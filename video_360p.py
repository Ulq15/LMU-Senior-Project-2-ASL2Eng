import os
import subprocess
import datetime
from enum import Enum
from tqdm import tqdm

# enumeration of video operations
class FFmpegOperatorEnum(Enum):
    Modify_Video_Resolution = 0
    Modify_Video_BitRate = 1
    Modify_Video_FrameRate = 2


class FFmpegBatchConversionVideo:
    m_TotalConversionFiles = 0
    m_TotalFiles = 0
    m_SupportVideoFormat = ['.mp4','.avi']
    m_FFmpegOperatorEnum = None

    m_Video_Resolution = ''
    m_Video_BitRate = ''
    m_Video_FrameRate = ''

    def __init__(self,videoformat = ['.mp4','.avi'],ffmpegOperatorEnum = FFmpegOperatorEnum.Modify_Video_Resolution):
        self.m_SupportVideoFormat = videoformat
        self.m_FFmpegOperatorEnum = ffmpegOperatorEnum
        pass

    def ConvertBatchVideos(self,inputPath,outputPath):
        if not os.path.isdir(outputPath):
            os.mkdir(outputPath)

        #for files in os.listdir(inputPath):
        for files in tqdm(os.listdir(inputPath)): #Add tqdm progress bar
            input_name = os.path.join(inputPath,files)
            output_name = os.path.join(outputPath,files)

            # If the input path is a file
            if os.path.isfile(input_name):
                dirPath = (os.path.abspath(os.path.dirname(output_name)))
                # Create the output folder if it doesn't exist
                if not os.path.isdir(dirPath):
                    os.mkdir(dirPath)
                # Determine whether the suffix name of the input video is in the supported list
                #if os.path.split(input_name)[-1].lower() in self.m_SupportVideoFormat:
                if input_name.split('.')[1].lower() in self.m_SupportVideoFormat:
                    # Modify video resolution
                    if self.m_FFmpegOperatorEnum == FFmpegOperatorEnum.Modify_Video_Resolution:
                        self.ModifyVideoResolution(input_name,output_name)
                    # Modify the video bitrate
                    elif self.m_FFmpegOperatorEnum == FFmpegOperatorEnum.Modify_Video_BitRate:
                        self.ModifyVideoBitRate(input_name,output_name)
                    # Modify the video frame rate
                    elif self.m_FFmpegOperatorEnum == FFmpegOperatorEnum.Modify_Video_FrameRate:
                        self.ModifyVideoFrameRate(input_name,output_name)
                    else:
                        pass
                self.m_TotalFiles += 1

            # If the input path is a folder
            else:
                # Create the output folder if it doesn't exist
                if not os.path.isdir(output_name):
                    os.mkdir(output_name)
                # recursion
                self.ConvertBatchVideos(input_name,output_name)

    def ModifyVideoResolution(self,videoin,videoout):
        t_ffmpegcmdline = 'ffmpeg -i "{}"  -vf scale={} -threads 4 "{}" -hide_banner'.format(videoin,self.m_Video_Resolution ,videoout)
        returncode = subprocess.call(t_ffmpegcmdline)
        self.m_TotalConversionFiles += 1

    def ModifyVideoBitRate(self,videoin,videoout):
        t_ffmpegcmdline = 'ffmpeg -i "{}"  -b:v {} -threads 4 "{}" -hide_banner'.format(videoin, self.m_Video_BitRate ,videoout)
        returncode = subprocess.call(t_ffmpegcmdline)
        self.m_TotalConversionFiles += 1

    def ModifyVideoFrameRate(self,videoin,videoout):
        t_ffmpegcmdline = 'ffmpeg -r {} -i "{}"  -threads 4 "{}" -hide_banner'.format(self.m_Video_FrameRate,videoin, videoout)
        returncode = subprocess.call(t_ffmpegcmdline)
        self.m_TotalConversionFiles += 1

if __name__ == '__main__':
    inputDir = r'input path'
    outputDir = r'output path'

    # record the total conversion time
    opeartion_start_time = datetime.datetime.now()

    # Batch modify video frame rate
    # ffmpegBatchConversionVideo = FFmpegBatchConversionVideo(['mp4','avi'],ffmpegOperatorEnum=FFmpegOperatorEnum.Modify_Video_FrameRate)
    # ffmpegBatchConversionVideo.m_Video_FrameRate = '60'
    # ffmpegBatchConversionVideo.ConvertBatchVideos(inputDir,outputDir)

    # Batch modify video bitrate
    # ffmpegBatchConversionVideo = FFmpegBatchConversionVideo(['mp4','avi'],ffmpegOperatorEnum=FFmpegOperatorEnum.Modify_Video_BitRate)
    # ffmpegBatchConversionVideo.m_Video_BitRate = '10000k'
    # ffmpegBatchConversionVideo.ConvertBatchVideos(inputDir,outputDir)

    # Batch modify video resolution
    ffmpegBatchConversionVideo = FFmpegBatchConversionVideo(['mp4','avi'],ffmpegOperatorEnum=FFmpegOperatorEnum.Modify_Video_Resolution)
    ffmpegBatchConversionVideo.m_Video_Resolution = '64:64'
    ffmpegBatchConversionVideo.ConvertBatchVideos(inputDir,outputDir)

    opeartion_end_time = datetime.datetime.now()
    opeartion_duration = opeartion_end_time - opeartion_start_time

    print('The conversion is completed, there are {} video files in total,'
          ' and {} video files are converted,'
          ' which takes a total of {}'.
          format(ffmpegBatchConversionVideo.m_TotalFiles,
          ffmpegBatchConversionVideo.m_TotalConversionFiles,
          opeartion_duration))
