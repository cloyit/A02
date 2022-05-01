package com.example.drive.controller;


import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.api.R;
import com.example.drive.entity.Picture;
import com.example.drive.entity.User;
import com.example.drive.mapper.PictureMapper;
import com.example.drive.response.RespBean;
import com.example.drive.service.IUserService;
import com.example.drive.service.impl.UserServiceImpl;
import com.example.drive.utills.FastDFSUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.Resource;
import java.io.IOException;
import java.time.LocalDateTime;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
@RestController
@RequestMapping("/user")
public class UserController {
    @Autowired
    IUserService userService;
    @Autowired
    PictureMapper pictureMapper;

    /**
     * 注册
     * @param u
     * @return
     */
    @PostMapping("/register")
    public RespBean register(@RequestBody User u){
        userService.register(u);
        return RespBean.ok("success It is suggested to improve personal information. The information that can be improved is as follows",u);
    }

    /**
     * 完善个人信息
     * @return
     */
    @PostMapping("perfectInformation")
    public RespBean perfectInformation(@RequestBody User u){
        userService.perfectInformation(u);
        return RespBean.ok("success",u);
    }

    /**
     *获取紧急电话号码
     * @return
     */
    @GetMapping("getEmergencyNumber")
    public RespBean getEmergencyNumber(){
        return RespBean.ok("Emergency contacts are as follows",userService.getEmergencyNumber());
    }

    /**
     * 上传头像
     * @param file
     * @return
     */
    @PostMapping("uploadPortrait")
    public RespBean uploadPortrait(MultipartFile file){
        Picture picture = new Picture();


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

        User user = userService.getUser();
        user.setHeadPortrait(picture.getHref());
        userService.perfectInformation(user);
        return RespBean.ok("success",picture);


    }
    /**
     *获取当前user的信息
     * @return
     */
    @GetMapping("getUser")
    public RespBean getUser(){
        return RespBean.ok("user information is",userService.getUser());
    }



    /**
     *绑定手环
     * @return
     */
    @PostMapping("bindBracelet")
    public RespBean bindBracelet(Integer bracelet){
        User u = userService.getUser();
        u.setBracelet(bracelet);
        //获取user ，更新 插入
        userService.perfectInformation(u);
        return RespBean.ok("user information is",userService.getUser());
    }



    /**
     *获取用户手环
     * @return
     */
    @GetMapping("getBracelet")
    public RespBean getBracelet(){
        User u = userService.getUser();
        int bracelet = u.getBracelet();
        if(bracelet==0){
            return RespBean.error("not band Bracelet");
        }
        return RespBean.ok("user bracelet id is",bracelet);
    }


    @GetMapping("getApp")
    public ResponseEntity<FileSystemResource> getApp() {
        String contentDisposition = ContentDisposition
                .builder("attachment")
                .filename("/home/zhulu02/anzhixing.apk")
                .build().toString();
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, contentDisposition)
                .contentType(MediaType.IMAGE_JPEG)
                .body(new FileSystemResource("/home/zhulu02/anzhixing.apk"));
    }




}
