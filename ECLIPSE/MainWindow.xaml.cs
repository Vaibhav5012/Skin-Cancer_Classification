using Microsoft.Win32;
using LiveCharts;
using LiveCharts.Wpf;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.ObjectModel;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;

namespace ECLIPSE_V3
{
    public class PredictionItem
    {
        public string Label { get; set; }
        public double Percentage { get; set; }
        public string PercentageText => $"{Percentage:0.##}%";
    }

    public partial class MainWindow : Window
    {
        private readonly string ModelRelative = @"Models\model.onnx";
        private OnnxModelRunner _runner;
        private string _imagePath;
        private float[] _lastScores;

        // full and short labels (HAM10000)
        private readonly string[] HAM10000Labels = new[]
        {
            "akiec - Actinic Keratoses",
            "bcc  - Basal Cell Carcinoma",
            "bkl  - Benign Keratosis-like",
            "df   - Dermatofibroma",
            "mel  - Melanoma",
            "nv   - Melanocytic Nevi",
            "vasc - Vascular Lesions"
        };

        private readonly string[] HAM10000ShortLabels = new[]
        {
            "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
        };

        // LiveCharts
        public SeriesCollection BarSeries { get; set; } = new SeriesCollection();
        public List<string> ChartLabels { get; set; } = new List<string>();
        public Func<double, string> PercentFormatter { get; set; } = v => $"{v:0.##}%";

        public ObservableCollection<PredictionItem> TopPredictions { get; } = new ObservableCollection<PredictionItem>();

        public MainWindow()
        {
            try
            {
                InitializeComponent();
            }
            catch (Exception ex)
            {
                MessageBox.Show("XAML load failed: " + ex.Message, "Startup error", MessageBoxButton.OK, MessageBoxImage.Error);
                throw;
            }

            BarChart.Series = BarSeries;
            DataContext = this;

            // ensure ChartLabels holds the short codes
            ChartLabels = HAM10000ShortLabels.ToList();
            if (BarChart.AxisX != null && BarChart.AxisX.Count > 0)
                BarChart.AxisX[0].Labels = ChartLabels;

            // keyboard shortcuts
            this.CommandBindings.Add(new CommandBinding(ApplicationCommands.Open, (s, e) => BtnLoadImage_Click(s, null)));

            // load model in background
            Task.Run(() => LoadModelAsync());
        }

        private void Log(string text)
        {
            Dispatcher.Invoke(() =>
            {
                if (TxtLog != null)
                {
                    TxtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {text}\n");
                    TxtLog.ScrollToEnd();
                }
            });
        }

        #region Assets
        private void Window_Loaded(object sender, RoutedEventArgs e) => LoadAssets();

        private void LoadAssets()
        {
            try
            {
                string exeDir = AppDomain.CurrentDomain.BaseDirectory;
                string assetsDir = Path.Combine(exeDir, "Assets");
                string bgPath = Path.Combine(assetsDir, "background.jpg");
                string bannerPath = Path.Combine(assetsDir, "banner.png");

                Log($"Looking for background: {bgPath}");
                if (File.Exists(bgPath))
                {
                    var bg = LoadBitmap(bgPath);
                    BackgroundImage.Source = bg;
                    BackgroundImage.Visibility = Visibility.Visible;
                }

                Log($"Looking for banner: {bannerPath}");
                if (File.Exists(bannerPath))
                {
                    var bmp = LoadBitmap(bannerPath);
                    var brush = new ImageBrush(bmp) { Stretch = Stretch.UniformToFill };
                    BannerRect.Fill = brush;
                    BannerRect.Visibility = Visibility.Visible;
                }
            }
            catch (Exception ex)
            {
                Log("Asset load error: " + ex.Message);
            }
        }

        private BitmapImage LoadBitmap(string path)
        {
            var bi = new BitmapImage();
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                bi.BeginInit();
                bi.CacheOption = BitmapCacheOption.OnLoad;
                bi.StreamSource = fs;
                bi.EndInit();
                bi.Freeze();
            }
            return bi;
        }
        #endregion

        #region Model load
        private void LoadModelAsync()
        {
            try
            {
                var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ModelRelative);
                if (!File.Exists(modelPath))
                {
                    Log("Model file not found: " + modelPath);
                    return;
                }

                var runner = new OnnxModelRunner(modelPath,
                    inputName: "input",
                    outputName: "output",
                    width: 224,
                    height: 224,
                    channelsFirst: true,
                    scaleToZeroOne: true
                );

                _runner = runner;
                Log("Model loaded from: " + modelPath);
                Dispatcher.Invoke(() => BtnPredict.IsEnabled = _imagePath != null);
            }
            catch (Exception ex)
            {
                Log("Model load failed: " + ex.ToString());
            }
        }
        #endregion

        #region Load / Drag & Drop
        private void BtnLoadImage_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog();
            dlg.Filter = "Image files|*.jpg;*.jpeg;*.png;*.bmp|All files|*.*";
            if (dlg.ShowDialog() != true) return;
            SetImagePath(dlg.FileName);
        }

        private void SetImagePath(string path)
        {
            _imagePath = path;
            try
            {
                var bmp = LoadBitmap(_imagePath);
                PreviewImage.Source = bmp;
                AnimateImageFade();

                BtnPredict.IsEnabled = _runner != null;
                Log("Image loaded: " + _imagePath);
            }
            catch (Exception ex)
            {
                Log("Preview load failed: " + ex.Message);
            }
        }

        private void PreviewArea_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop)) DragHint.Visibility = Visibility.Visible;
        }

        private void PreviewArea_DragLeave(object sender, DragEventArgs e)
        {
            DragHint.Visibility = Visibility.Collapsed;
        }

        private void PreviewArea_Drop(object sender, DragEventArgs e)
        {
            DragHint.Visibility = Visibility.Collapsed;
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var files = (string[])e.Data.GetData(DataFormats.FileDrop);
                var f = files.FirstOrDefault();
                if (f != null && (f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)))
                {
                    SetImagePath(f);
                }
                else
                {
                    MessageBox.Show("Please drop an image file.", "Drag & Drop", MessageBoxButton.OK, MessageBoxImage.Warning);
                }
            }
        }

        private void AnimateImageFade()
        {
            var a = new DoubleAnimation(0, 1, TimeSpan.FromMilliseconds(280));
            PreviewImage.BeginAnimation(System.Windows.UIElement.OpacityProperty, a);
        }
        #endregion

        #region Predict + charts + top3
        private async void BtnPredict_Click(object sender, RoutedEventArgs e)
        {
            if (_runner == null)
            {
                MessageBox.Show("Model not loaded.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (string.IsNullOrEmpty(_imagePath) || !File.Exists(_imagePath))
            {
                MessageBox.Show("Load an image first.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            PredictProgress.Visibility = Visibility.Visible;
            BtnPredict.IsEnabled = false;
            BtnLoadImage.IsEnabled = false;
            BtnExport.IsEnabled = false;

            Log("Running prediction...");
            try
            {
                var scores = await Task.Run(() => _runner.PredictFromImage(_imagePath));
                if (scores == null || scores.Length == 0)
                {
                    Log("Model returned no output.");
                    MessageBox.Show("Model returned no output.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                _lastScores = scores;
                var probs = Softmax(scores);
                UpdateChartsAndList(probs);
                Log("Prediction finished.");
            }
            catch (Exception ex)
            {
                Log("Prediction failed: " + ex.Message);
                MessageBox.Show("Prediction failed: " + ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                PredictProgress.Visibility = Visibility.Collapsed;
                BtnPredict.IsEnabled = true;
                BtnLoadImage.IsEnabled = true;
                BtnExport.IsEnabled = true;
            }
        }

        private static double[] Softmax(float[] scores)
        {
            var max = scores.Max();
            var exps = scores.Select(s => Math.Exp(s - max)).ToArray();
            var sum = exps.Sum();
            return exps.Select(e => e / sum).ToArray();
        }

        private void UpdateChartsAndList(double[] probs)
        {
            Dispatcher.Invoke(() =>
            {
                // fixed number of classes (7)
                int n = HAM10000ShortLabels.Length;

                // ensure probability array has n items (pad with zeros if needed)
                double[] pcts = new double[n];
                for (int i = 0; i < n; i++)
                {
                    double v = (probs != null && i < probs.Length) ? probs[i] : 0.0;
                    pcts[i] = Math.Round(v * 100.0, 2);
                }

                // --- BAR CHART: ensure correct values and axis step = 1 ---
                BarSeries.Clear();

                // column series with all values
                var col = new ColumnSeries
                {
                    Title = "Probability",
                    Values = new ChartValues<double>(pcts),
                    MaxColumnWidth = 48,
                    DataLabels = false
                };
                BarSeries.Add(col);

                // set short labels and force step=1 so every label/column is shown
                ChartLabels = HAM10000ShortLabels.Take(n).ToList();
                if (BarChart.AxisX != null && BarChart.AxisX.Count > 0)
                {
                    var axisX = BarChart.AxisX[0];
                    axisX.Labels = ChartLabels;
                    // Force tick step to 1 (one label per column). Hide separators if desired.
                    axisX.Separator = new Separator { Step = 1, IsEnabled = false };
                    // make labels rotate slightly if they collide
                    axisX.LabelsRotation = (ChartLabels.Count > 6) ? 10 : 0;
                }

                // --- PIE chart (full labels) ---
                PieChart.Series.Clear();
                for (int i = 0; i < n; i++)
                {
                    PieChart.Series.Add(new PieSeries
                    {
                        Title = HAM10000Labels[i],
                        Values = new ChartValues<double> { pcts[i] },
                        DataLabels = false,
                        PushOut = (pcts[i] == pcts.Max()) ? 8 : 0
                    });
                }

                // --- Top-3 ---
                TopPredictions.Clear();
                var top = Enumerable.Range(0, n)
                    .Select(i => new { idx = i, pct = pcts[i], label = HAM10000Labels[i] })
                    .OrderByDescending(x => x.pct)
                    .Take(3)
                    .ToList();

                foreach (var t in top)
                    TopPredictions.Add(new PredictionItem { Label = t.label, Percentage = t.pct });

                // small entrance animation
                var da = new DoubleAnimation(0, 1, TimeSpan.FromMilliseconds(350));
                BarChart.BeginAnimation(OpacityProperty, da);
                PieChart.BeginAnimation(OpacityProperty, da);
            });
        }

        #endregion

        #region CSV export
        private void BtnExport_Click(object sender, RoutedEventArgs e)
        {
            if (_lastScores == null || _lastScores.Length == 0)
            {
                MessageBox.Show("No prediction to export.", "Export", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var sfd = new SaveFileDialog { Filter = "CSV file|*.csv", FileName = $"prediction_{DateTime.Now:yyyyMMdd_HHmmss}.csv" };
            if (sfd.ShowDialog() != true) return;

            try
            {
                // Prepare metadata
                string predictionId = Guid.NewGuid().ToString();
                string timestamp = DateTime.Now.ToString("o");
                string imagePathSafe = _imagePath ?? "";
                long imageFileSize = 0;
                try
                {
                    if (!string.IsNullOrEmpty(_imagePath) && File.Exists(_imagePath))
                    {
                        imageFileSize = new FileInfo(_imagePath).Length;
                    }
                }
                catch { /* ignore file access issues, leave size 0 */ }

                // model info
                string modelFullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ModelRelative);
                bool modelPresent = File.Exists(modelFullPath);

                // compute softmax probabilities and ranks
                var probs = Softmax(_lastScores);
                int n = Math.Min(HAM10000Labels.Length, Math.Max(probs.Length, HAM10000ShortLabels.Length));

                // create ranking: higher prob => lower rank number (1 = top)
                var order = Enumerable.Range(0, n).OrderByDescending(i => (i < probs.Length ? probs[i] : 0.0)).ToList();
                var rankMap = new Dictionary<int, int>();
                for (int r = 0; r < order.Count; r++)
                    rankMap[order[r]] = r + 1;

                using (var sw = new StreamWriter(sfd.FileName))
                {
                    // header
                    sw.WriteLine("prediction_id,timestamp,image_path,image_filesize_bytes,class_full,class_short,logit,probability,percentage,rank,is_top1,is_top3,model_path,model_present");

                    for (int i = 0; i < n; i++)
                    {
                        // safe extraction
                        double logit = (i < _lastScores.Length) ? _lastScores[i] : 0.0;
                        double probability = (i < probs.Length) ? probs[i] : 0.0;
                        double percentage = Math.Round(probability * 100.0, 2);
                        int rank = rankMap.ContainsKey(i) ? rankMap[i] : (n + 1);
                        bool isTop1 = rank == 1;
                        bool isTop3 = rank <= 3;

                        // safe quoting for fields that may contain commas or special chars
                        string classFull = (i < HAM10000Labels.Length) ? HAM10000Labels[i] : $"class_{i}";
                        string classShort = (i < HAM10000ShortLabels.Length) ? HAM10000ShortLabels[i] : $"c{i}";
                        string quote(string s) => "\"" + (s ?? "").Replace("\"", "\"\"") + "\"";

                        // use invariant culture for numbers
                        var logitStr = logit.ToString("G6", System.Globalization.CultureInfo.InvariantCulture);
                        var probStr = probability.ToString("0.######", System.Globalization.CultureInfo.InvariantCulture);
                        var pctStr = percentage.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture) + "%";

                        sw.WriteLine(
                            $"{predictionId}," +
                            $"{timestamp}," +
                            $"{quote(imagePathSafe)}," +
                            $"{imageFileSize}," +
                            $"{quote(classFull)}," +
                            $"{classShort}," +
                            $"{logitStr}," +
                            $"{probStr}," +
                            $"{quote(pctStr)}," +
                            $"{rank}," +
                            $"{isTop1.ToString().ToLower()}," +
                            $"{isTop3.ToString().ToLower()}," +
                            $"{quote(modelFullPath)}," +
                            $"{modelPresent.ToString().ToLower()}"
                        );
                    }
                }

                Log("Exported CSV to: " + sfd.FileName);
                Process.Start("explorer.exe", $"/select,\"{sfd.FileName}\"");
            }
            catch (Exception ex)
            {
                MessageBox.Show("Export failed: " + ex.Message, "Export", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        #endregion


        #region Misc UI
        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.O && (Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control) BtnLoadImage_Click(sender, null);
            if (e.Key == Key.P && (Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control) BtnPredict_Click(sender, null);
            if (e.Key == Key.S && (Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control) BtnExport_Click(sender, null);
        }

        private void BtnClearLog_Click(object sender, RoutedEventArgs e) => TxtLog.Clear();
        #endregion
    }
}
