package com.example.drive.utills;

import ch.qos.logback.classic.Logger;
import org.csource.common.MyException;
import org.csource.fastdfs.*;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class FastDFSUtil {
    private static Logger logger = (Logger) LoggerFactory.getLogger(FastDFSUtil.class);
    /**
     * 文件上传
     */
    public static String [] upload(byte[] buffFile,String fileExtName) {
        TrackerServer ts=null;
        StorageServer ss=null;
        try {
            //连接远程服务器
            //读取FastDFS的配置文件用于将所有的tracker的地址读取到内存中
            ClientGlobal.init("fastdfs.conf");
            TrackerClient tc=new TrackerClient();
            ts=tc.getConnection();
            ss=tc.getStoreStorage(ts);
            //定义Storage的客户端对象，需要使用这个对象来完成具体的文件上传 下载和删除操作
            StorageClient sc=new StorageClient(ts,ss);
            /**
             * 文件上传
             * 参数 1 为需要上传的文件的字节数组
             * 参数 2 为需要上传的文件的扩展名
             * 参数 3 为文件的属性文件通常不上传
             * 返回一个String数组 这个数据对我们非常总要必须妥善保管建议存入数据库
             * 数组中的第一个元素为文件所在的组名
             * 数组中的第二个元素为文件所在远程路径名
             */
            String[] result= sc.upload_file(buffFile,fileExtName,null);
            return result;
        } catch (IOException e) {
            logger.error("发生错误",e);
        } catch (MyException e) {
            logger.error("发生错误",e);
        } finally {
            if(ss!=null){
                try {
                    ss.close();
                } catch (IOException e) {
                    logger.error("发生错误",e);
                }
            }
            if(ts!=null){
                try {
                    ts.close();
                } catch (IOException e) {
                    logger.error("发生错误",e);
                }
            }
        }
        return null;
    }

    /**
     * 下载文件
     */
    public static byte [] download(String groupName,String remoteFilename) {
        TrackerServer ts=null;
        StorageServer ss=null;
        try {
            //读取FastDFS的配置文件用于将所有的tracker的地址读取到内存中
            ClientGlobal.init("fastdfs.conf");
            TrackerClient tc=new TrackerClient();
            ts=tc.getConnection();
            ss=tc.getStoreStorage(ts);
            //定义Storage的客户端对象，需要使用这个对象来完成具体的文件上传 下载和删除操作
            StorageClient sc=new StorageClient(ts,ss);
            /**
             * 文件下载
             * 参数1 需要下载的文件的组名
             * 参数2 需要下载文件的远程文件名
             */
            byte [] buffFile=sc.download_file(groupName,remoteFilename);
            return buffFile;
        } catch (IOException e) {
            logger.error("发生错误",e);
        } catch (MyException e) {
            logger.error("发生错误",e);
        } finally {
            if(ss!=null){
                try {
                    ss.close();
                } catch (IOException e) {
                    logger.error("发生错误",e);
                }
            }
            if(ts!=null){
                try {
                    ts.close();
                } catch (IOException e) {
                    logger.error("发生错误",e);
                }
            }
        }

        return null;
    }

    /**
     * 文件删除
     * @return
     */
    public static int delete(String groupName, String remoteFilename) {
        TrackerServer ts=null;
        StorageServer ss=null;
        int result = 1;
        try {
            //读取FastDFS的配置文件用于将所有的tracker的地址读取到内存中
            ClientGlobal.init("fastdfs.conf");
            TrackerClient tc=new TrackerClient();
            ts=tc.getConnection();
            ss=tc.getStoreStorage(ts);
            //定义Storage的客户端对象，需要使用这个对象来完成具体的文件上传 下载和删除操作
            StorageClient sc=new StorageClient(ts,ss);
            /**
             * 文件删除
             * 参数1 需要删除的文件的组名
             * 参数2 需要删除文件的远程文件名
             * 返回一个int类型的数据 返回0 表示文件删除成功其它值表示文件在删除失败
             */
            result=sc.delete_file(groupName,remoteFilename);
        } catch (IOException e) {
            logger.error("发生错误",e);
        } catch (MyException e) {
            logger.error("发生错误",e);
        } finally {
            if(ss!=null){
                try {
                    ss.close();
                } catch (IOException e) {
                    logger.error("发生错误",e);
                }
            }
            if(ts!=null){
                try {
                    ts.close();
                } catch (IOException e) {
                    logger.error("发生错误",e);
                }
            }
            return result;
        }
    }

    //将文件转换成Byte数组
    public static byte[] getBytesByFile(File file) {
        try {
            FileInputStream fis = new FileInputStream(file);
            ByteArrayOutputStream bos = new ByteArrayOutputStream(1000);
            byte[] b = new byte[1000];
            int n;
            while ((n = fis.read(b)) != -1) {
                bos.write(b, 0, n);
            }
            fis.close();
            byte[] data = bos.toByteArray();
            bos.close();
            return data;
        } catch (Exception e) {
            logger.error("发生错误",e);
        }
        return null;
    }

}
