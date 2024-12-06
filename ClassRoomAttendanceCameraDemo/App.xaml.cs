using System.Configuration;
using System.Data;
using System.Windows;
using ClassRoomAttendanceCameraDemo.ViewModel;
using Mvvm.Common.WindowExtensions;

namespace ClassRoomAttendanceCameraDemo
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        /// <summary>
        ///   引发 <see cref="E:System.Windows.Application.Startup" /> 事件。
        /// </summary>
        /// <param name="e">
        ///   包含事件数据的 <see cref="T:System.Windows.StartupEventArgs" />。
        /// </param>
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            var model = new MainWindowViewModel();
            model.CreateWindow<MainWindow>().Show();
        }
    }

}
