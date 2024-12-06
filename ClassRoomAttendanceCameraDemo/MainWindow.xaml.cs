using System.Collections.Generic;
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ClassRoomAttendanceCameraDemo.ViewModel;
using Mvvm.Common.WindowExtensions;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenVinoSharp;
using Point = OpenCvSharp.Point;
using Rect = OpenCvSharp.Rect;
using Window = System.Windows.Window;
using Path=System.IO.Path;

namespace ClassRoomAttendanceCameraDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

    }
}