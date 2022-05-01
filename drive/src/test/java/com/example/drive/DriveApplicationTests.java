package com.example.drive;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.drive.entity.*;
import com.example.drive.mapper.*;
import com.example.drive.service.IDrivingInformationService;
import com.example.drive.service.IUserService;
import com.example.drive.utills.FastDFSUtil;
import org.apache.tomcat.jni.Local;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.io.File;
import java.sql.Date;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Random;

@SpringBootTest
class DriveApplicationTests {
    @Autowired
    UserMapper userMapper;

    @Autowired
    PeachMapper peachMapper;
    @Autowired
    PasswordEncoder passwordEncoder;
    @Autowired
    TitleMapper titleMapper;
    @Autowired
    CarMapper carMapper;
    @Autowired
    DetailMapper detailMapper;
    @Autowired
    BrandMapper brandMapper;
    @Autowired
    UserHealthMapper userHealthMapper;
    @Autowired
    IDrivingInformationService iDrivingInformationService;
    @Autowired
    IUserService iUserService;
    @Autowired
    DrivingInformationMapper drivingInformationMapper;
    @Test
    void contextLoads() {

   DrivingInformation drivingInformation = null;

   LocalDateTime beginTime;
        LocalDateTime endTime;
   Random e =new Random();
   for(int i=0;i<27;i++){
       drivingInformation = new DrivingInformation();
       beginTime =LocalDateTime.of(2022,1,i+1,12,24);
       endTime = LocalDateTime.of(2022,1,i+1,12+e.nextInt(4),20+e.nextInt(30));
       drivingInformation.setAttention(2+e.nextInt(3));
       drivingInformation.setUid(1473527123812581377L);
       drivingInformation.setBegin(beginTime);
       drivingInformation.setEnd(endTime);
       drivingInformation.setCarId(3);
       drivingInformation.setCloseEye(1+e.nextInt(3));
       drivingInformation.setEndLatitude(31+e.nextInt(2)+"");
       drivingInformation.setStartLatitude(31+e.nextInt(2)+"");
       drivingInformation.setStartLongitude(31+e.nextInt(2)+"");
       drivingInformation.setEndLongitude(31+e.nextInt(2)+"");
       drivingInformationMapper.insert(drivingInformation);
   }



    }

    @Test
    void test1() {


        List<DrivingInformation> drivingInformationList = drivingInformationMapper.selectList(null);
        for (DrivingInformation d : drivingInformationList) {
            Random r = new Random();
            d.setStartLatitude("114." + r.nextInt(40));
            d.setEndLatitude("115." + r.nextInt(40));
            d.setStartLongitude("30." + r.nextInt(40));
            d.setEndLongitude("31." + r.nextInt(40));
            QueryWrapper<DrivingInformation> queryWrapper = new QueryWrapper<>();
            queryWrapper.eq("id", d.getId());
            drivingInformationMapper.update(d, queryWrapper);
        }
    }
        @Test
        void test2(){
        userHealthMapper.delete(null);


        }

    @Test
    void test3() throws InterruptedException {


        while (true) {
            LocalDateTime localDateTime = LocalDateTime.now();
            UserHealth d = new UserHealth();

            Random r = new Random();
            int wendu = 5 + r.nextInt(2);
            d.setHeartbeat(80+r.nextInt(10)+"");
            d.setBloodOxygen(83+r.nextInt(5)+"");
            d.setTemperature("35."+r.nextInt(9));
            d.setTime(localDateTime);
            userHealthMapper.insert(d);
            System.out.println(d.getId());
            Thread.sleep(5000);
        }


    }

}
