// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using System.Globalization;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cpu;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;

namespace RFDETRNanoDemo
{
    /// <summary>
    /// Demonstrates RF-DETR Nano object detection on a static image.
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static DetectionDrawingOptions _detectionDrawingOptions = default!;
        private static SegmentationDrawingOptions _segmentationDrawingOptions = default!;

        static void Main(string[] args)
        {
            CreateOutputFolder();
            SetDrawingOptions();

            var runSegmentation = true;
            var modelPath = Path.Join(SharedConfig.ModelsFolder, runSegmentation ? "rfdetr-stain.onnx" : "rfdetr-nano.onnx");
            var classNamesPath = Path.Join(SharedConfig.ModelsFolder, runSegmentation? "rfdetr-stain-class_names.txt" : "rfdetr-nano-class_names.txt");

            if (EnsureDemoAssets(modelPath, classNamesPath, runSegmentation) is false)
                return;

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(
                    model: modelPath,
                    modelType: runSegmentation ? ModelType.Segmentation : ModelType.ObjectDetection,
                    modelVersion: ModelVersion.RFDETR,
                    labels: LoadLabels(classNamesPath)),

                ImageResize = ImageResize.Stretched,
                SamplingOptions = new(SKFilterMode.Linear, SKMipmapMode.None),

                // RF-DETR uses ImageNet normalization after scaling RGB values to 0..1.
                ImageMean = [0.485f, 0.456f, 0.406f],
                ImageStd = [0.229f, 0.224f, 0.225f]
            });

            Console.WriteLine($"Loaded ONNX Model: {yolo.ModelInfo}");

            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.ObjectDetection));

            if (runSegmentation)
                RunSegmentationTest(yolo, image);
            else
                RunObjectDetectionTest(yolo, image);

            DisplayOutputFolder();
        }

        private static void RunObjectDetectionTest(Yolo yolo, SKBitmap image)
        {
            var results = yolo.RunObjectDetection(image, confidence: 0.4, iou: 0.7);

            image.Draw(results, _detectionDrawingOptions);

            var fileName = Path.Combine(_outputFolder, "RFDETRNanoObjectDetection.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);

            PrintResults(results);
        }

        private static void RunSegmentationTest(Yolo yolo, SKBitmap image)
        {
            var results = yolo.RunSegmentation(image, confidence: 0.4, pixelConfedence: 0.5, iou: 0.7);

            image.Draw(results, _segmentationDrawingOptions);

            var fileName = Path.Combine(_outputFolder, "RFDETRNanoSegmentation.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);

            PrintResults(results);
        }

        private static string[] LoadLabels(string path)
            => File.ReadAllLines(path)
                .Where(label => string.IsNullOrWhiteSpace(label) is false)
                .ToArray();

        private static bool EnsureDemoAssets(string modelPath, string classNamesPath, bool segmentation)
        {
            if (File.Exists(modelPath) && File.Exists(classNamesPath))
                return true;

            var mode = segmentation ? "segmentation" : "object detection";

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"RF-DETR Nano {mode} demo assets are missing.");
            Console.WriteLine($"Expected model: {modelPath}");
            Console.WriteLine($"Expected labels: {classNamesPath}");
            Console.WriteLine("Run detection asset download with: python Demo/RFDETRNanoDemo/download-rfdetr-nano.py");
            Console.WriteLine("For segmentation, place an RF-DETR segmentation ONNX model at test/assets/Models/rfdetr-nano-seg.onnx.");
            Console.ForegroundColor = ConsoleColor.Gray;

            return false;
        }

        private static void SetDrawingOptions()
        {
            _detectionDrawingOptions = new DetectionDrawingOptions
            {
                DrawBoundingBoxes = true,
                DrawConfidenceScore = true,
                DrawLabels = true,
                EnableFontShadow = true,
                Font = SKTypeface.Default,
                FontSize = 18,
                FontColor = SKColors.White,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
                BorderThickness = 2,
                BoundingBoxOpacity = 128,
            };

            _segmentationDrawingOptions = new SegmentationDrawingOptions
            {
                DrawBoundingBoxes = true,
                DrawConfidenceScore = true,
                DrawLabels = true,
                EnableFontShadow = true,
                Font = SKTypeface.Default,
                FontSize = 18,
                FontColor = SKColors.White,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
                BorderThickness = 2,
                BoundingBoxOpacity = 128,
                DrawSegmentationPixelMask = true,
                PixelMaskOpacity = 128,
                DrawContour = true,
                ContourThickness = 2,
            };
        }

        private static void PrintResults(List<ObjectDetection> results)
        {
            Console.WriteLine();
            Console.WriteLine($"Inference Results: {results.Count} objects");
            Console.WriteLine(new string('=', 80));

            Console.ForegroundColor = ConsoleColor.Blue;

            foreach (var result in results)
            {
                var label = result.Label.Name;
                var confidence = (result.Confidence * 100).ToString("0.##", CultureInfo.InvariantCulture);
                Console.WriteLine($"{label} ({confidence}%)");
            }

            Console.ForegroundColor = ConsoleColor.Gray;
        }

        private static void PrintResults(List<Segmentation> results)
        {
            Console.WriteLine();
            Console.WriteLine($"Inference Results: {results.Count} segmentations");
            Console.WriteLine(new string('=', 80));

            Console.ForegroundColor = ConsoleColor.Blue;

            foreach (var result in results)
            {
                var label = result.Label.Name;
                var confidence = (result.Confidence * 100).ToString("0.##", CultureInfo.InvariantCulture);
                Console.WriteLine($"{label} ({confidence}%) maskBytes={result.BitPackedPixelMask.Length}");
            }

            Console.ForegroundColor = ConsoleColor.Gray;
        }

        private static void CreateOutputFolder()
        {
            _outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");

            if (Directory.Exists(_outputFolder) is false)
                Directory.CreateDirectory(_outputFolder);
        }

        private static void DisplayOutputFolder()
        {
            var shell = OperatingSystem.IsWindows() ? "explorer"
                     : OperatingSystem.IsLinux() ? "xdg-open"
                     : OperatingSystem.IsMacOS() ? "open"
                     : null;

            if (shell is not null)
                Process.Start(shell, _outputFolder);
            else
                Console.WriteLine($"Results saved to: {_outputFolder}");
        }
    }
}
