import asyncio
import json
import time
from typing import (AsyncGenerator, AsyncIterator, Callable, Dict, Final, List,
                    Optional)
from typing import Sequence as GenericSequence
from typing import Union

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb, ChatCompletionLogProbs,
    ChatCompletionLogProbsContent, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ErrorResponse, FunctionCall, PromptTokenUsageInfo,
    RequestResponseMetadata, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (BaseModelPath,
                                                    LoRAModulePath,
                                                    OpenAIServing,
                                                    PromptAdapterPath)
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizers import maybe_serialize_tool_calls
from vllm.utils import iterate_with_cancellation

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        base_model_paths: List[BaseModelPath],
        response_role: str,
        *,
        lora_modules: Optional[List[LoRAModulePath]],
        prompt_adapters: Optional[List[PromptAdapterPath]],
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        enable_auto_tools: bool = False,
        tool_parser: Optional[str] = None,
        enable_prompt_tokens_details: bool = False,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=lora_modules,
                         prompt_adapters=prompt_adapters,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                "\"auto\" tool choice has been enabled please note that while"
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored.")

        self.tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None
        if self.enable_auto_tools:
            try:
                if (tool_parser == "pythonic" and
                        model_config.model.startswith("meta-llama/Llama-3.2")):
                    logger.warning(
                        "Llama3.2 models may struggle to emit valid pythonic"
                        " tool calls")
                self.tool_parser = ToolParserManager.get_tool_parser(
                    tool_parser)
            except Exception as e:
                raise TypeError("Error: --enable-auto-tool-choice requires "
                                f"tool_parser:'{tool_parser}' which has not "
                                "been registered") from e

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        
    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]
        
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ErrorResponse]:
        """Simplified streaming-only chat completion."""
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            tokenizer = await self.engine_client.get_tokenizer()

            conversation, request_prompts, engine_prompts = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=request.chat_template or self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
            )

            sampling_params = request.to_sampling_params(
                self.max_model_len - len(engine_prompts[0]["prompt_token_ids"])
            )

            generator = self.engine_client.generate(
                engine_prompts[0],  # We know there's only one
                sampling_params,
                request_id=f"chatcmpl-{request.request_id}",
                priority=request.priority,
            )

            if raw_request:
                generator = iterate_with_cancellation(
                    generator, raw_request.is_disconnected)

            return self.chat_completion_stream_generator(
                request, generator, "", None, tokenizer, None)

        except ValueError as e:
            return self.create_error_response(str(e))
            
    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[Dict[int, Logprob]]],
        tokenizer: AnyTokenizer,
        num_output_top_logprobs: Optional[int] = None,
    ) -> ChatCompletionLogProbs:
        logprobs_content: List[ChatCompletionLogProbsContent] = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                continue
            
            step_token = step_top_logprobs[token_id]
            logprobs_content.append(
                ChatCompletionLogProbsContent(
                    token=self._get_decoded_token(
                        step_token,
                        token_id,
                        tokenizer,
                        self.return_tokens_as_token_ids,
                    ),
                    logprob=max(step_token.logprob, -9999.0)
                )
            )

        return ChatCompletionLogProbs(content=logprobs_content)

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        first_iteration = True

        try:
            async for res in result_generator:
                if first_iteration:
                    role = self.get_chat_request_role(request)
                    choice_data = ChatCompletionResponseStreamChoice(
                        delta=DeltaMessage(
                            role=role,  # Add role here
                            content="",
                        ),
                        logprobs=None,
                        token_ids=[0]
                    )
                    chunk = ChatCompletionStreamResponse(choices=[choice_data])
                    response_json = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {response_json}\n\n"
                    first_iteration = False
                    
                for output in res.outputs:
                    if not output.text and not output.token_ids:
                        continue

                    if request.logprobs and output.logprobs is not None:
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                        )
                    else:
                        logprobs = None

                    choice_data = ChatCompletionResponseStreamChoice(
                        delta=DeltaMessage(content=output.text),
                        logprobs=logprobs,
                        token_ids=list(output.token_ids)
                    )

                    chunk = ChatCompletionStreamResponse(choices=[choice_data])
                    response_json = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {response_json}\n\n"

        except ValueError as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"