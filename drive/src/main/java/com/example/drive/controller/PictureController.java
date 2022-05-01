package com.example.drive.controller;


import com.example.drive.entity.Picture;
import com.example.drive.mapper.PictureMapper;
import com.example.drive.response.RespBean;
import com.example.drive.utills.FastDFSUtil;
import com.google.zxing.BarcodeFormat;
import com.google.zxing.EncodeHintType;
import com.google.zxing.MultiFormatWriter;
import com.google.zxing.WriterException;
import com.google.zxing.client.j2se.MatrixToImageConfig;
import com.google.zxing.client.j2se.MatrixToImageWriter;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zhulu
 * @since 2022-03-02
 */
@RestController
@RequestMapping("/picture")
public class PictureController {
    @Autowired
    PictureMapper pictureMapper;

    @PostMapping("uploadPicture")
    public RespBean uploadPicture(MultipartFile file,String description){
        Picture picture = new Picture();
        picture.setDescription(description);


        String[] result = null;
        int num = pictureMapper.selectCount(null);

        try {
            result = FastDFSUtil.upload(file.getBytes(),""+(num+1)+".jpg");
        } catch (IOException e) {
            e.printStackTrace();
        }
        picture.setUrl("http://47.102.99.215/"+result[0]+"/"+result[1]);
        picture.setHref("http://47.102.99.215/"+result[0]+"/"+result[1]);

        pictureMapper.insert(picture);

        return RespBean.ok("success",picture);


    }


}
