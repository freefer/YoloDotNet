// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.RFDETR
{
    internal class SegmentationModuleRFDETR : ISegmentationModule
    {
        private const string BackgroundClassPrefix = "background_class";

        private readonly YoloCore _yoloCore;
        private readonly int _predictions;
        private readonly int _totalLabels;
        private readonly int _modelInputWidth;
        private readonly int _modelInputHeight;
        private readonly int _maskWidth;
        private readonly int _maskHeight;
        private readonly int _maskPlaneSize;
        private readonly int _backgroundClassIndex;
        private readonly bool _hasBackgroundClass;
        private readonly LabelModel[] _labels;
        private int[] _topIndices = [];
        private float[] _topScores = [];
        private List<Segmentation> _results = default!;

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleRFDETR(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            var inputShape = _yoloCore.OnnxModel.InputShapes.ElementAt(0).Value;
            var detsShape = _yoloCore.OnnxModel.OutputShapes["dets"];
            var labelsShape = _yoloCore.OnnxModel.OutputShapes["labels"];
            var masksShape = _yoloCore.OnnxModel.OutputShapes["masks"];

            _modelInputHeight = (int)inputShape[2];
            _modelInputWidth = (int)inputShape[3];
            _predictions = detsShape[1];
            _totalLabels = labelsShape[2];
            _maskHeight = masksShape[2];
            _maskWidth = masksShape[3];
            _maskPlaneSize = _maskWidth * _maskHeight;
            _backgroundClassIndex = Array.FindIndex(
                _yoloCore.OnnxModel.Labels,
                label => label.Name.StartsWith(BackgroundClassPrefix, StringComparison.OrdinalIgnoreCase));
            _hasBackgroundClass = _backgroundClassIndex >= 0;
            _labels = _hasBackgroundClass
                ? [.. _yoloCore.OnnxModel.Labels
                    .Where(label => label.Index != _backgroundClassIndex)
                    .Select((label, index) => label with { Index = index })]
                : _yoloCore.OnnxModel.Labels;
            _results = [];
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
        {
            var inferenceResult = _yoloCore.Run(image, roi);
            var resultBuffer = ArrayPool<ObjectResult>.Shared.Rent(_predictions);

            try
            {
                var count = RunSegmentation(inferenceResult, confidence, resultBuffer);
                return YoloCore.InferenceResultsToType(resultBuffer.AsSpan(0, count), roi, _results, r => (Segmentation)r);
            }
            finally
            {
                ArrayPool<ObjectResult>.Shared.Return(resultBuffer, false);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int RunSegmentation(InferenceResult inferenceResult, double confidenceThreshold, ObjectResult[] resultBuffer)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var boxesSpan = inferenceResult.OrtSpan0;
            var logitsSpan = inferenceResult.OrtSpan1;
            var masksSpan = inferenceResult.OrtSpan2;

            if (boxesSpan.IsEmpty || logitsSpan.IsEmpty || masksSpan.IsEmpty)
                return 0;

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);
            var (topIndices, topScores) = GetTopKBuffers(_predictions);
            var topCount = GetRankedCandidates(logitsSpan, topIndices, topScores);
            var validBoxCount = 0;

            for (var candidateIndex = 0; candidateIndex < topCount; candidateIndex++)
            {
                var candidateConfidence = topScores[candidateIndex];

                if (candidateConfidence < confidenceThreshold)
                    continue;

                var flatIndex = topIndices[candidateIndex];
                var prediction = flatIndex / _totalLabels;
                var labelIndex = flatIndex - prediction * _totalLabels;

                if (_hasBackgroundClass && labelIndex == _backgroundClassIndex)
                    continue;

                var mappedLabelIndex = MapLabelIndex(labelIndex);

                if (mappedLabelIndex < 0 || mappedLabelIndex >= _labels.Length)
                    continue;

                var boxOffset = prediction * 4;
                var cx = boxesSpan[boxOffset];
                var cy = boxesSpan[boxOffset + 1];
                var w = boxesSpan[boxOffset + 2];
                var h = boxesSpan[boxOffset + 3];

                var xMin = cx - w * 0.5f;
                var yMin = cy - h * 0.5f;
                var xMax = cx + w * 0.5f;
                var yMax = cy + h * 0.5f;

                SKRectI boundingBox;
                SKRect boundingBoxUnscaled;

                if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                {
                    var scaledXMin = xMin * _modelInputWidth - xPad;
                    var scaledYMin = yMin * _modelInputHeight - yPad;
                    var scaledXMax = xMax * _modelInputWidth - xPad;
                    var scaledYMax = yMax * _modelInputHeight - yPad;

                    boundingBox = ClampBox(
                        scaledXMin * xGain,
                        scaledYMin * xGain,
                        scaledXMax * xGain,
                        scaledYMax * xGain,
                        imageSize);
                    boundingBoxUnscaled = new SKRect(scaledXMin, scaledYMin, scaledXMax, scaledYMax);
                }
                else
                {
                    boundingBox = ClampBox(
                        xMin * imageSize.Width,
                        yMin * imageSize.Height,
                        xMax * imageSize.Width,
                        yMax * imageSize.Height,
                        imageSize);
                    boundingBoxUnscaled = new SKRect(
                        xMin * _modelInputWidth,
                        yMin * _modelInputHeight,
                        xMax * _modelInputWidth,
                        yMax * _modelInputHeight);
                }

                if (boundingBox.Width <= 0 || boundingBox.Height <= 0)
                    continue;

                resultBuffer[validBoxCount++] = new ObjectResult
                {
                    Label = _labels[mappedLabelIndex],
                    Confidence = candidateConfidence,
                    BoundingBox = boundingBox,
                    BoundingBoxUnscaled = boundingBoxUnscaled,
                    BoundingBoxIndex = prediction,
                    BitPackedPixelMask = PackMask(masksSpan, prediction * _maskPlaneSize, boundingBox, imageSize)
                };
            }

            return validBoxCount;
        }

        private int GetRankedCandidates(ReadOnlySpan<float> logitsSpan, int[] topIndices, float[] topScores)
        {
            var topCount = 0;
            var minScore = float.PositiveInfinity;
            var minPosition = -1;

            for (var prediction = 0; prediction < _predictions; prediction++)
            {
                var labelOffset = prediction * _totalLabels;

                for (var labelIndex = 0; labelIndex < _totalLabels; labelIndex++)
                {
                    var confidence = YoloCore.Sigmoid(logitsSpan[labelOffset + labelIndex]);
                    var flatIndex = labelOffset + labelIndex;

                    if (topCount < _predictions)
                    {
                        topIndices[topCount] = flatIndex;
                        topScores[topCount] = confidence;

                        if (confidence < minScore)
                        {
                            minScore = confidence;
                            minPosition = topCount;
                        }

                        topCount++;
                        continue;
                    }

                    if (confidence <= minScore)
                        continue;

                    topIndices[minPosition] = flatIndex;
                    topScores[minPosition] = confidence;
                    minScore = topScores[0];
                    minPosition = 0;

                    for (var i = 1; i < topCount; i++)
                    {
                        var candidateScore = topScores[i];

                        if (candidateScore < minScore)
                        {
                            minScore = candidateScore;
                            minPosition = i;
                        }
                    }
                }
            }

            SortTopKDescending(topIndices, topScores, topCount);

            return topCount;
        }

        private byte[] PackMask(ReadOnlySpan<float> masksSpan, int maskOffset, SKRectI box, SKSizeI imageSize)
        {
            var targetWidth = box.Width;
            var targetHeight = box.Height;

            if (targetWidth <= 0 || targetHeight <= 0 || _maskWidth <= 0 || _maskHeight <= 0)
                return [];

            var totalPixels = targetWidth * targetHeight;
            var packed = new byte[(totalPixels + 7) / 8];
            var cropLeft = 0;
            var cropTop = 0;
            var cropRight = _maskWidth;
            var cropBottom = _maskHeight;

            if (_yoloCore.YoloOptions.ImageResize != ImageResize.Stretched)
            {
                var scale = MathF.Min((float)_modelInputWidth / imageSize.Width, (float)_modelInputHeight / imageSize.Height);
                var scaledWidth = (int)(imageSize.Width * scale);
                var scaledHeight = (int)(imageSize.Height * scale);
                var padX = (_modelInputWidth - scaledWidth) * 0.5f;
                var padY = (_modelInputHeight - scaledHeight) * 0.5f;

                cropLeft = Math.Clamp((int)MathF.Round(padX * _maskWidth / _modelInputWidth), 0, _maskWidth - 1);
                cropTop = Math.Clamp((int)MathF.Round(padY * _maskHeight / _modelInputHeight), 0, _maskHeight - 1);
                cropRight = Math.Clamp((int)MathF.Round((padX + scaledWidth) * _maskWidth / _modelInputWidth), cropLeft + 1, _maskWidth);
                cropBottom = Math.Clamp((int)MathF.Round((padY + scaledHeight) * _maskHeight / _modelInputHeight), cropTop + 1, _maskHeight);
            }

            var cropWidth = cropRight - cropLeft;
            var cropHeight = cropBottom - cropTop;
            var xScale = (float)cropWidth / imageSize.Width;
            var yScale = (float)cropHeight / imageSize.Height;

            for (var y = 0; y < targetHeight; y++)
            {
                var absoluteY = box.Top + y;
                var sourceY = (absoluteY + 0.5f) * yScale - 0.5f;
                var y0Local = Math.Clamp((int)MathF.Floor(sourceY), 0, cropHeight - 1);
                var y1Local = y0Local < cropHeight - 1 ? y0Local + 1 : y0Local;
                var yWeight = sourceY - y0Local;
                var row0 = maskOffset + (cropTop + y0Local) * _maskWidth;
                var row1 = maskOffset + (cropTop + y1Local) * _maskWidth;
                var targetRow = y * targetWidth;

                for (var x = 0; x < targetWidth; x++)
                {
                    var absoluteX = box.Left + x;
                    var sourceX = (absoluteX + 0.5f) * xScale - 0.5f;
                    var x0Local = Math.Clamp((int)MathF.Floor(sourceX), 0, cropWidth - 1);
                    var x1Local = x0Local < cropWidth - 1 ? x0Local + 1 : x0Local;
                    var xWeight = sourceX - x0Local;
                    var x0 = cropLeft + x0Local;
                    var x1 = cropLeft + x1Local;
                    var topLeft = masksSpan[row0 + x0];
                    var topRight = masksSpan[row0 + x1];
                    var bottomLeft = masksSpan[row1 + x0];
                    var bottomRight = masksSpan[row1 + x1];
                    var top = topLeft + (topRight - topLeft) * xWeight;
                    var bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;

                    // Official RF-DETR inference thresholds masks at > 0 after optional resize/crop.
                    if (top + (bottom - top) * yWeight > 0)
                    {
                        var pixelIndex = targetRow + x;
                        packed[pixelIndex >> 3] |= (byte)(1 << (pixelIndex & 0b0111));
                    }
                }
            }

            return packed;
        }

        private (int[] TopIndices, float[] TopScores) GetTopKBuffers(int size)
        {
            if (_topIndices.Length < size)
            {
                _topIndices = new int[size];
                _topScores = new float[size];
            }

            return (_topIndices, _topScores);
        }

        private static void SortTopKDescending(int[] indices, float[] scores, int length)
        {
            for (var i = 1; i < length; i++)
            {
                var score = scores[i];
                var index = indices[i];
                var j = i - 1;

                while (j >= 0 && scores[j] < score)
                {
                    scores[j + 1] = scores[j];
                    indices[j + 1] = indices[j];
                    j--;
                }

                scores[j + 1] = score;
                indices[j + 1] = index;
            }
        }

        private int MapLabelIndex(int labelIndex)
        {
            if (!_hasBackgroundClass)
                return labelIndex;

            return labelIndex > _backgroundClassIndex
                ? labelIndex - 1
                : labelIndex;
        }

        private static SKRectI ClampBox(float xMin, float yMin, float xMax, float yMax, SKSizeI imageSize)
        {
            var left = Math.Clamp((int)xMin, 0, imageSize.Width - 1);
            var top = Math.Clamp((int)yMin, 0, imageSize.Height - 1);
            var right = Math.Clamp((int)xMax, 0, imageSize.Width - 1);
            var bottom = Math.Clamp((int)yMax, 0, imageSize.Height - 1);

            return new SKRectI(left, top, right, bottom);
        }

        public void Dispose()
        {
            _yoloCore?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
