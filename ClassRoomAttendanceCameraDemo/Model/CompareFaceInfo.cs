using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace ClassRoomAttendanceCameraDemo.Model
{
    /// <summary>
    /// 比对人脸信息
    /// <para>bb 2024-03-04 15:13:11</para>
    /// </summary>
    public struct CompareFaceInfo
    {
        /// <summary>
        /// 头像
        /// </summary>
        public Mat Head { get; set; }
        /// <summary>
        /// 人脸底库
        /// </summary>
        public Mat FaceBase { get; set; }
        /// <summary>
        /// 是否匹配成功
        /// </summary>
        public bool IsMatchSuccess { get; set; }
        /// <summary>
        /// 名称
        /// </summary>
        public string Name { get; set; }
        /// <summary>
        /// Id
        /// </summary>
        public string Id { get; set; }
    }
}
