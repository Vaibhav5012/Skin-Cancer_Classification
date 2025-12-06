using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace ECLIPSE
{
    public class OnnxModelRunner : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly string _outputName;
        private readonly int _width;
        private readonly int _height;
        private readonly bool _channelsFirst;
        private readonly bool _scaleToZeroOne;

        public OnnxModelRunner(string modelPath, string inputName = null, string outputName = null, int width = 224, int height = 224, bool channelsFirst = true, bool scaleToZeroOne = true)
        {
            _session = new InferenceSession(modelPath);
            _inputName = inputName ?? _session.InputMetadata.Keys.First();
            _outputName = outputName ?? _session.OutputMetadata.Keys.First();
            _width = width;
            _height = height;
            _channelsFirst = channelsFirst;
            _scaleToZeroOne = scaleToZeroOne;
        }

        public float[] PredictFromImage(string imagePath)
        {
            using (var bmp = new Bitmap(imagePath))
            {
                using (var resized = Resize(bmp, _width, _height))
                {
                    var input = ImageToTensor(resized);
                    var container = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(_inputName, input)
                    };

                    using (var results = _session.Run(container))
                    {
                        var first = results.First();
                        // Accept multiple numeric types - try to extract float[]
                        if (first.Value is Tensor<float> tf)
                        {
                            return tf.ToArray();
                        }
                        // try float[] directly
                        if (first.Value is System.Array arr)
                        {
                            var floats = new float[arr.Length];
                            for (int i = 0; i < arr.Length; i++)
                            {
                                floats[i] = Convert.ToSingle(arr.GetValue(i));
                            }
                            return floats;
                        }

                        // fallback: try to convert via enumerable
                        try
                        {
                            return ((dynamic)first.Value).ToArray();
                        }
                        catch
                        {
                            return null;
                        }
                    }
                }
            }
        }

        private Tensor<float> ImageToTensor(Bitmap bmp)
        {
            // channelsFirst: [1, C, H, W]
            int channels = 3;
            if (_channelsFirst)
            {
                var tensor = new DenseTensor<float>(new[] { 1, channels, _height, _width });
                for (int y = 0; y < _height; y++)
                {
                    for (int x = 0; x < _width; x++)
                    {
                        var px = bmp.GetPixel(x, y);
                        // normalize to [0,1] or keep 0-255
                        if (_scaleToZeroOne)
                        {
                            tensor[0, 0, y, x] = px.R / 255f;
                            tensor[0, 1, y, x] = px.G / 255f;
                            tensor[0, 2, y, x] = px.B / 255f;
                        }
                        else
                        {
                            tensor[0, 0, y, x] = px.R;
                            tensor[0, 1, y, x] = px.G;
                            tensor[0, 2, y, x] = px.B;
                        }
                    }
                }
                return tensor;
            }
            else
            {
                var tensor = new DenseTensor<float>(new[] { 1, _height, _width, channels });
                for (int y = 0; y < _height; y++)
                {
                    for (int x = 0; x < _width; x++)
                    {
                        var px = bmp.GetPixel(x, y);
                        if (_scaleToZeroOne)
                        {
                            tensor[0, y, x, 0] = px.R / 255f;
                            tensor[0, y, x, 1] = px.G / 255f;
                            tensor[0, y, x, 2] = px.B / 255f;
                        }
                        else
                        {
                            tensor[0, y, x, 0] = px.R;
                            tensor[0, y, x, 1] = px.G;
                            tensor[0, y, x, 2] = px.B;
                        }
                    }
                }
                return tensor;
            }
        }

        private Bitmap Resize(Bitmap source, int width, int height)
        {
            var dest = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(dest))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(source, 0, 0, width, height);
            }
            return dest;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
