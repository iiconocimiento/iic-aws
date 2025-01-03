import io
import json
from typing import Optional

from sagemaker.huggingface.model import HuggingFacePredictor


class LineIterator:
    """
    A helper class for parsing the byte stream input.

    The output of the model will be in the following format:
    ```
    b'{"outputs": [" a"]}\\n'
    b'{"outputs": [" challenging"]}\\n'
    b'{"outputs": [" problem"]}\\n'
    ...
    ```

    While usually each PayloadPart event from the event stream will contain a byte array
    with a full json, this is not guaranteed and some of the json objects may be split across
    PayloadPart events. For example:
    ```
    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\\n'}}
    ```

    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\\n' character) within
    the buffer via the 'scan_lines' function. It maintains the position of the last read
    position to ensure that previous bytes are not exposed again.

    Source: https://github.com/aws-samples/amazon-sagemaker-llama2-response-streaming-recipes/blob/695327e5436e04e5161c4061fa71c8b553ed6609/llama-2-hf-tgi/llama-2-7b-chat-hf/utils/LineIterator.py
    """

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                if b"{" in line and (line := line[line.find(b"{") : -1]):
                    return line
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


class TextGenerationPredictor(HuggingFacePredictor):
    """A predictor class for Text Generation models."""

    def predict(
        self,
        data,
        initial_args=None,
        target_model=None,
        target_variant=None,
        inference_id=None,
        custom_attributes=None,
        component_name: Optional[str] = None,
    ):
        """
        Return the inference from the specified endpoint.

        Args:
            data (object): Input data for which you want the model to provide
                inference. If a serializer was specified when creating the
                Predictor, the result of the serializer is sent as input
                data. Otherwise the data must be sequence of bytes, and the
                predict method then sends the bytes in the request body as is.
            initial_args (dict[str,str]): Optional. Default arguments for boto3
                ``invoke_endpoint`` call. Default is None (no default
                arguments).
            target_model (str): S3 model artifact path to run an inference request on,
                in case of a multi model endpoint. Does not apply to endpoints hosting
                single model (Default: None)
            target_variant (str): The name of the production variant to run an inference
                request on (Default: None). Note that the ProductionVariant identifies the
                model you want to host and the resources you want to deploy for hosting it.
            inference_id (str): If you provide a value, it is added to the captured data
                when you enable data capture on the endpoint (Default: None).
            custom_attributes (str): Provides additional information about a request for an
                inference submitted to a model hosted at an Amazon SageMaker endpoint.
                The information is an opaque value that is forwarded verbatim. You could use this
                value, for example, to provide an ID that you can use to track a request or to
                provide other metadata that a service endpoint was programmed to process. The value
                must consist of no more than 1024 visible US-ASCII characters.

                The code in your model is responsible for setting or updating any custom attributes
                in the response. If your code does not set this value in the response, an empty
                value is returned. For example, if a custom attribute represents the trace ID, your
                model can prepend the custom attribute with Trace ID: in your post-processing
                function (Default: None).
            component_name (str): Optional. Name of the Amazon SageMaker inference component
                corresponding the predictor.

        Returns:
            object: Inference for the given input. If a deserializer was specified when creating
                the Predictor, the result of the deserializer is
                returned. Otherwise the response returns the sequence of bytes
                as is.
        """
        data |= {"stream": False}
        response = super(TextGenerationPredictor, self).predict(
            data,
            initial_args,
            target_model,
            target_variant,
            inference_id,
            custom_attributes,
            component_name,
        )
        if isinstance(response, list):
            response = response[0]
        if isinstance(response, dict):
            response.pop("model", None)
        return response

    def stream(
        self,
        data,
        initial_args=None,
        target_model=None,
        target_variant=None,
        inference_id=None,
        custom_attributes=None,
        component_name: Optional[str] = None,
    ):
        """
        Stream the inference response from the specified endpoint.

        Args:
            data (object): Input data for which you want the model to provide
                inference. If a serializer was specified when creating the
                Predictor, the result of the serializer is sent as input
                data. Otherwise the data must be sequence of bytes, and the
                predict method then sends the bytes in the request body as is.
            initial_args (dict[str,str]): Optional. Default arguments for boto3
                ``invoke_endpoint`` call. Default is None (no default
                arguments).
            target_model (str): S3 model artifact path to run an inference request on,
                in case of a multi model endpoint. Does not apply to endpoints hosting
                single model (Default: None)
            target_variant (str): The name of the production variant to run an inference
                request on (Default: None). Note that the ProductionVariant identifies the
                model you want to host and the resources you want to deploy for hosting it.
            inference_id (str): If you provide a value, it is added to the captured data
                when you enable data capture on the endpoint (Default: None).
            custom_attributes (str): Provides additional information about a request for an
                inference submitted to a model hosted at an Amazon SageMaker endpoint.
                The information is an opaque value that is forwarded verbatim. You could use this
                value, for example, to provide an ID that you can use to track a request or to
                provide other metadata that a service endpoint was programmed to process. The value
                must consist of no more than 1024 visible US-ASCII characters.

                The code in your model is responsible for setting or updating any custom attributes
                in the response. If your code does not set this value in the response, an empty
                value is returned. For example, if a custom attribute represents the trace ID, your
                model can prepend the custom attribute with Trace ID: in your post-processing
                function (Default: None).
            component_name (str): Optional. Name of the Amazon SageMaker inference component
                corresponding the predictor.

        Returns:
        -------
            Iterator[dict]: Chunks of the response as sent back from the endpoint.
        """
        data |= {"stream": True}
        data["parameters"] = data.get("parameters", {}) | {"return_full_text": False}

        request_args = self._create_request_args(
            data=data,
            initial_args=initial_args,
            target_model=target_model,
            target_variant=target_variant,
            inference_id=inference_id,
            custom_attributes=custom_attributes,
        )

        inference_component_name = component_name or self._get_component_name()
        if inference_component_name:
            request_args["InferenceComponentName"] = inference_component_name

        response = self.sagemaker_session.sagemaker_runtime_client.invoke_endpoint_with_response_stream(
            **request_args
        )
        for line in LineIterator(response["Body"]):
            chunk = json.loads(line.decode("utf-8"))
            if isinstance(response, dict):
                chunk.pop("model", None)
            yield chunk
