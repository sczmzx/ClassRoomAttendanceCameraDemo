using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Documents;
using System.Windows.Input;
using ClassRoomAttendanceCameraDemo.Model;
using Mvvm.Common;
using Mvvm.Common.Commands;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenVinoSharp;
using OpenVinoSharp.Extensions.process;
using OpenVinoSharp.Extensions.result;

namespace ClassRoomAttendanceCameraDemo.ViewModel
{
    public class MainWindowViewModel:ViewModelBase,IViewModel,IModelLoaded
    {
        private string _inputAddress; //输入地址

        /// <summary>
        /// 输入地址
        /// </summary>
        public string InputAddress
        {
            get => _inputAddress;
            set => SetProperty(ref _inputAddress, value);
        }

        private string _recognModeAddress; //模型地址

        /// <summary>
        /// 模型地址
        /// </summary>
        public string RecognModeAddress
        {
            get => _recognModeAddress;
            set =>SetProperty( ref _recognModeAddress, value);
        }

        private string _compareModeAddress; //模型地址

        /// <summary>
        /// 模型地址
        /// </summary>
        public string CompareModeAddress
        {
            get => _compareModeAddress;
            set =>SetProperty( ref _compareModeAddress, value);
        }
        private string _faceBaseDir; //人脸地库文件夹

        /// <summary>
        /// 人脸地库文件夹
        /// </summary>
        public string FaceBaseDir
        {
            get => _faceBaseDir;
            set =>SetProperty(ref _faceBaseDir, value);
        }

        private ObservableCollection<string> _deviceCollection; //设备列表

        /// <summary>
        /// 设备列表
        /// </summary>
        public ObservableCollection<string> DeviceCollection
        {
            get => _deviceCollection;
            set => SetProperty(ref _deviceCollection, value);
        }

        private string _selectDevice; //选择的设备

        /// <summary>
        /// 选择的设备
        /// </summary>
        public string SelectDevice
        {
            get => _selectDevice;
            set => SetProperty(ref _selectDevice, value);
        }
        private FlowDocument _document;

        public FlowDocument Document
        {
            get => _document;
            set => SetProperty(ref _document, value);
        }
        private Core _core;
        public ICommand StartReadCommand { get; set; }


        public MainWindowViewModel()
        {
            InputAddress = "rtsp://admin:coursedev2019@10.6.30.87:554/video1";
            CompareModeAddress = "facenet_inception_resnetv1.onnx";
            RecognModeAddress= "yolov8n-face-lindevs.onnx";
            FaceBaseDir = "FaceBase";
            DeviceCollection = new ObservableCollection<string>();
            StartReadCommand = new DelegateCommand(StartRead);
            Document = new FlowDocument();
        }

        /// <summary>
        /// 添加文本
        /// </summary>
        /// <param name="text"></param>
        private void AddLineText(string text)
        {
            Document.Dispatcher.Invoke(() =>
            {
                Document.Blocks.Add(new Paragraph(new Run(text)));
            });
        }

        private InferRequest _recognRequest;
        private InferRequest _compareRequest;

        /// <summary>
        /// 初始化模型
        /// </summary>
        /// <returns></returns>
        private Task InitRequest()
        {
            Task recognTask=Task.Run(() =>
            {
                _recognRequest = GetRequest(RecognModeAddress);
            });
            Task compareTask=Task.Run(() =>
            {
                _compareRequest = GetRequest(CompareModeAddress);
            });
            return Task.WhenAll(recognTask, compareTask);
        }
        /// <summary>
        /// 开始识别
        /// </summary>
        private async void StartRead()
        {
            Document.Blocks.Clear();
            Cv2.DestroyAllWindows();
            Stopwatch sw = Stopwatch.StartNew();
            var inputImage = GetInputImage();
            if (inputImage.Empty())
            {
                AddLineText($"获取输入图片失败");
                return;
            }
            sw.Stop();
            AddLineText($"获取输入图片耗时:{sw.Elapsed.TotalMilliseconds}ms");
            //获取模型耗时：

            sw.Restart();
            await InitRequest();
            var request = _recognRequest;
            sw.Stop();
            AddLineText($"初始化模型耗时({SelectDevice}):{sw.Elapsed.TotalMilliseconds}ms");
            sw.Restart();
            var headImages = ImagePredict(request, inputImage).ToList();
            sw.Stop();
            AddLineText($"识别人脸人数：{headImages.Count}; 耗时:{sw.Elapsed.TotalMilliseconds}ms");
            foreach (var compareInfo in CompareFaces(headImages))
            {
                if (compareInfo.IsMatchSuccess)
                {
                    Cv2.ImShow(compareInfo.Name, MergeImagesHorizontally(compareInfo.Head, compareInfo.FaceBase));
                }
            }
        }

        static Mat MergeImagesHorizontally(Mat img1, Mat img2)
        {
            // 获取两张图片的宽度和高度
            int width1 = img1.Cols;
            int height1 = img1.Rows;
            int width2 = img2.Cols;
            int height2 = img2.Rows;

            // 计算合并后图片的宽度和高度
            int mergedWidth = width1 + width2;
            int mergedHeight = Math.Max(height1, height2);

            // 创建一个新的图像矩阵
            Mat mergedImage = new Mat(mergedHeight, mergedWidth, img1.Type());

            // 将第一张图片复制到新图像的左半部分
            img1.CopyTo(mergedImage[new Rect(0, 0, width1, height1)]);

            // 将第二张图片复制到新图像的右半部分
            img2.CopyTo(mergedImage[new Rect(width1, 0, width2, height2)]);

            return mergedImage;
        }
        /// <summary>
        /// 获取request
        /// </summary>
        /// <param name="modeAddress"></param>
        /// <returns></returns>
        private InferRequest GetRequest(string modeAddress)
        {
            var model = _core.read_model(modeAddress);
            var compiledModel = _core.compile_model(model, SelectDevice);
            var request = compiledModel.create_infer_request();
            return request;
        }

        /// <summary>
        /// 人脸比对
        /// </summary>
        /// <param name="headImages"></param>
        /// <returns></returns>
        private IEnumerable<CompareFaceInfo> CompareFaces(IEnumerable<Mat> headImages)
        {
            Stopwatch sw = Stopwatch.StartNew();
            var request = _compareRequest;
            //获取人脸底库特征码
            var faceBaseEmbeddings = GetFaceBaseEmbeddings(request).ToList();
            sw.Stop();
            AddLineText($"识别底库人脸人数： {faceBaseEmbeddings.Count} ; 耗时:{sw.Elapsed.TotalMilliseconds}ms");
            sw.Restart();
            //获取头像特征码
            var headEmbeddings = GetFaceEmbeddings(request, headImages);
            sw.Stop();
            AddLineText($"获取头像特征码人数：{headEmbeddings.Count} ；耗时:{sw.Elapsed.TotalMilliseconds}ms");
            sw.Restart();
            //将每个头像的特征码和底库的特征码做对比
            List<CompareFaceInfo> list = new List<CompareFaceInfo>();
            Parallel.ForEach(headEmbeddings, headEmbedding =>
            {
                var compareFaceInfo = new CompareFaceInfo();
                compareFaceInfo.Head = headEmbedding.Item1;
                compareFaceInfo.IsMatchSuccess = false;
                foreach (var faceBaseEmbedding in faceBaseEmbeddings)
                {
                    var distance = CompareFaces(headEmbedding.Item2, faceBaseEmbedding.Item2);
                    if (distance > 0.8)
                    {
                        compareFaceInfo.IsMatchSuccess = true;
                        compareFaceInfo.FaceBase = faceBaseEmbedding.Item3;
                        UpdateFaceName(ref compareFaceInfo, faceBaseEmbedding.Item1);
                        break;
                    }
                }
                lock (list)
                {
                    list.Add(compareFaceInfo);
                }
            });

            sw.Stop();
            AddLineText($"特征码对比人数：{list.Count} 成功人数：{list.Count(m=>m.IsMatchSuccess)} 总耗时:{sw.Elapsed.TotalMilliseconds}ms");
            return list;
        }

        /// <summary>
        /// 更新人员名单
        /// </summary>
        /// <param name="faceInfo"></param>
        /// <param name="fileName"></param>
        private void UpdateFaceName(ref CompareFaceInfo faceInfo, string fileName)
        {
            var split = fileName.Split('_');
            faceInfo.Name = split[0];
            if (split.Length > 1)
            {
                faceInfo.Id = split[1];
            }
        }

        private Dictionary<string, (bool,float[], Mat)> _dict = new Dictionary<string, (bool, float[], Mat)>();

        private IEnumerable<(string, float[], Mat)> GetFaceBaseEmbeddings(InferRequest request)
        {
            var faceBaseDir = FaceBaseDir;
            var files = System.IO.Directory.GetFiles(faceBaseDir);
            InferRequest recognRequest = _recognRequest;
            foreach (var file in files)
            {
                var fileName = Path.GetFileNameWithoutExtension(file);
                var isGet = _dict.TryGetValue(fileName, out var pair);
                if (!isGet)
               // if (true)
                {
                    var mat = new Mat(file);
                    //先将底库的人脸抓取出来
                    var headImages = ImagePredict(recognRequest, mat).ToList();
                    if (headImages.Count != 1)
                    {
                        _dict[fileName] = (false, default, default);
                    }
                    else
                    {
                        var data = GetFaceEmbedding(request, mat, out var categ_nums, out var factor);
                        _dict[fileName] = (true, data, mat);
                        yield return (fileName, data, mat);
                    }

                }
                else
                {
                    if (pair.Item1)
                    {
                        yield return (fileName, pair.Item2, pair.Item3);
                    }
                }
            }
        }

        private List<(Mat,float[])> GetFaceEmbeddings(InferRequest request, IEnumerable<Mat> headImages)
        {
            var result = new List<(Mat,float[])>();
            foreach (var headImage in headImages)
            {
                var data = GetFaceEmbedding(request, headImage, out var categ_nums, out var factor);
                result.Add((headImage,data));
            }
            return result;
        }
        /// <summary>
        /// 获取输入图片
        /// </summary>
        /// <returns></returns>
        private Mat GetInputImage()
        {
            var path = InputAddress;
            if (!File.Exists(path))
            {
                using VideoCapture videoCapture = new VideoCapture(path);
                if (!videoCapture.IsOpened())
                {
                    return new Mat();
                }
                var frame = new Mat();
                videoCapture.Read(frame);
                return frame;
            }
            else
            {
                return new Mat(path);
            }
        }

        public void Loaded()
        {
            _core = new Core();
            _core.get_available_devices().ForEach(DeviceCollection.Add);
            if (DeviceCollection.Any())
            {
                SelectDevice = DeviceCollection.First();
            }
        }

        IEnumerable<Mat> ImagePredict(InferRequest request, Mat image)
        {
            var data = GetFaceEmbedding(request, image, out var categ_nums, out var factor);
            DetResult result = PostProcess(data, categ_nums,
                factor);

           // Mat result_mat = image.Clone();
            for (int i = 0; i < result.count; i++)
            {
                yield return image[result.datas[i].box];
                //Cv2.Rectangle(result_mat, result.datas[i].box, new Scalar(0.0, 0.0, 255.0), 2);
                /*Cv2.Rectangle(result_mat, new Point(result.datas[i].box.TopLeft.X, result.datas[i].box.TopLeft.Y + 30), new Point(result.datas[i].box.BottomRight.X, result.datas[i].box.TopLeft.Y), new Scalar(0.0, 255.0, 255.0), -1);
                Cv2.PutText(result_mat, classes[result.datas[i].index] + "-" + result.datas[i].score.ToString("0.00"), new Point(result.datas[i].box.X, result.datas[i].box.Y + 25), HersheyFonts.HersheySimplex, 0.8, new Scalar(0.0, 0.0, 0.0), 2);*/
            }
           // Cv2.ImShow("predict",result_mat);
        }


        DetResult PostProcess(float[] result, int categ_nums, float factor) 
        {
            Mat result_data =Mat.FromPixelData(4 + categ_nums, 8400, MatType.CV_32F,result);
            result_data = result_data.T();

            // Storage results list
            List<Rect> position_boxes = new List<Rect>();
            List<int> classIds = new List<int>();
            List<float> confidences = new List<float>();
            // Preprocessing output results
            for (int i = 0; i < result_data.Rows; i++)
            {
                Mat classesScores = new Mat(result_data, new Rect(4, i, categ_nums, 1));
                Point maxClassIdPoint, minClassIdPoint;
                double maxScore, minScore;
                // Obtain the maximum value and its position in a set of data
                Cv2.MinMaxLoc(classesScores, out minScore, out maxScore,
                    out minClassIdPoint, out maxClassIdPoint);
                // Confidence level between 0 ~ 1
                // Obtain identification box information
                if (maxScore > 0.25)
                {
                    float cx = result_data.At<float>(i, 0);
                    float cy = result_data.At<float>(i, 1);
                    float ow = result_data.At<float>(i, 2);
                    float oh = result_data.At<float>(i, 3);
                    int x = (int)((cx - 0.5 * ow) * factor);
                    int y = (int)((cy - 0.5 * oh) * factor);
                    int width = (int)(ow * factor);
                    int height = (int)(oh * factor);
                    Rect box = new Rect();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;

                    position_boxes.Add(box);
                    classIds.Add(maxClassIdPoint.X);
                    confidences.Add((float)maxScore);
                }
            }
            // NMS non maximum suppression
            int[] indexes = new int[position_boxes.Count];
            float score = 0.25f;
            float nms = 0.45f;
            CvDnn.NMSBoxes(position_boxes, confidences, score, nms, out indexes);
            DetResult re = new DetResult();
            // 
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                re.add(classIds[index], confidences[index], position_boxes[index]);
            }
            return re;
        }
        public float[] GetFaceEmbedding(InferRequest request, Mat image,out int categNums,out float factor)
        {
            Tensor input_tensor = request.get_input_tensor();
            Shape input_shape = input_tensor.get_shape();
             factor = 0f;

            Mat mat = new Mat();
            Cv2.CvtColor(image, mat, ColorConversionCodes.BGR2RGB);
            mat = Resize.letterbox_img(mat, (int)input_shape[2], out factor);
            mat = Normalize.run(mat, true);
            float[] input_data = Permute.run(mat);
            input_tensor.set_data(input_data);

            request.infer();

            Tensor output_tensor = request.get_output_tensor();
            float[] output_data = output_tensor.get_data<float>((int)output_tensor.get_size());
            Shape output_shape = output_tensor.get_shape();
            categNums = (int)output_shape[1] - 4;
            return output_data;
        }

        /*public float CompareFaces(float[] embedding1, float[] embedding2)
        {
            if (embedding1.Length != embedding2.Length)
            {
                throw new ArgumentException("Embedding vectors must be of the same length.");
            }

            float sum = 0;
            for (int i = 0; i < embedding1.Length; i++)
            {
                sum += (embedding1[i] - embedding2[i]) * (embedding1[i] - embedding2[i]);
            }

            return (float)Math.Sqrt(sum);
        }*/
        static float CompareFaces(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of the same length.");
            }

            float dotProduct = 0.0f;
            float magnitude1 = 0.0f;
            float magnitude2 = 0.0f;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                magnitude1 += vector1[i] * vector1[i];
                magnitude2 += vector2[i] * vector2[i];
            }

            if (magnitude1 == 0.0f || magnitude2 == 0.0f)
            {
                return 0.0f;
            }

            return dotProduct / (float)(Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
        }
    }
}
