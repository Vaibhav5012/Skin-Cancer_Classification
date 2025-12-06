using Microsoft.Win32;
using System.Windows;

namespace ECLIPSE_V2
{
    public partial class SettingsWindow : Window
    {
        public string ModelPath { get; set; }
        public string BannerPath { get; set; }
        public string BackgroundPath { get; set; }
        public int InputWidth { get; set; } = 224;
        public int InputHeight { get; set; } = 224;

        public SettingsWindow()
        {
            InitializeComponent();
            TxtModel.Text = ModelPath;
            TxtBanner.Text = BannerPath;
            TxtBackground.Text = BackgroundPath;
        }

        private void BtnBrowseModel_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Filter = "ONNX model|*.onnx|All files|*.*" };
            if (dlg.ShowDialog() == true) TxtModel.Text = dlg.FileName;
        }

        private void BtnBrowseBanner_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Filter = "Image files|*.png;*.jpg;*.jpeg;*.bmp" };
            if (dlg.ShowDialog() == true) TxtBanner.Text = dlg.FileName;
        }

        private void BtnBrowseBackground_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Filter = "Image files|*.png;*.jpg;*.jpeg;*.bmp" };
            if (dlg.ShowDialog() == true) TxtBackground.Text = dlg.FileName;
        }

        private void BtnOk_Click(object sender, RoutedEventArgs e)
        {
            ModelPath = TxtModel.Text;
            BannerPath = TxtBanner.Text;
            BackgroundPath = TxtBackground.Text;
            DialogResult = true;
            Close();
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
