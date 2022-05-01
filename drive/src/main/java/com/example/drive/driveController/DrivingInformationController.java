package com.example.drive.driveController;


import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.drive.entity.Dpicture;
import com.example.drive.entity.DrivingInformation;
import com.example.drive.entity.Picture;
import com.example.drive.entity.UserHealth;
import com.example.drive.mapper.DpictureMapper;
import com.example.drive.response.RespBean;
import com.example.drive.service.IDrivingInformationService;
import com.example.drive.service.IUserService;
import com.example.drive.utills.FastDFSUtil;
import lombok.AllArgsConstructor;
import org.json.JSONException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
@RestController
@RequestMapping("/drivingInformation")
public class DrivingInformationController {
    @Autowired
    private IDrivingInformationService iDrivingInformationService;
    @Autowired
    private IUserService iUserService;
    @Autowired
    private DpictureMapper dpictureMapper;


    /**
     * 查看一段时间的驾驶记录
     * @return
     */
    @GetMapping("getDrivingInformationByDay")
    public RespBean getDrivingInformationByDay(String beginTimeS)  {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        LocalDateTime beginTime = LocalDateTime.parse(beginTimeS, formatter);
        List<DrivingInformation> DriveList = iDrivingInformationService.getDrivingInformationByDay(iUserService.getUid(),beginTime);
        if(DriveList!=null && DriveList.size()!=0){

            List<List<DrivingInformation>> result = new ArrayList<List<DrivingInformation>>();
            //拆分数据,设置标识，循环遍历
            int flag =0;
            for(int i =0;i<DriveList.size();i++){
                if(flag == DriveList.get(i).getBegin().getDayOfYear()){
                    result.get(result.size()-1).add(DriveList.get(i));
                }else {
                    //新建并且更新flag
                    flag = DriveList.get(i).getBegin().getDayOfYear();
                    List<DrivingInformation> r = new ArrayList<DrivingInformation>();
                    r.add(DriveList.get(i));
                    result.add(r);
                }
            }
            return RespBean.ok("DrivingInformation are ",result);
        }else {
            return RespBean.error("is empty");
        }

    }

    /**
     * 查看一段时间的驾驶记录
     * @return
     */
    @GetMapping("getDrivingInformationByTime")
    public RespBean getDrivingInformationByDay(String beginTimeS,String endTimeS)  {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        LocalDateTime beginTime = LocalDateTime.parse(beginTimeS, formatter);
        LocalDateTime endTime = LocalDateTime.parse(endTimeS, formatter);
        List<DrivingInformation> DriveList = iDrivingInformationService.getDrivingInformationByDay(iUserService.getUid(),beginTime,endTime);
        if(DriveList!=null && DriveList.size()!=0){

            List<List<DrivingInformation>> result = new ArrayList<List<DrivingInformation>>();
            //拆分数据,设置标识，循环遍历
            int flag =0;
            for(int i =0;i<DriveList.size();i++){
                if(flag == DriveList.get(i).getBegin().getDayOfYear()){
                    result.get(result.size()-1).add(DriveList.get(i));
                }else {
                    //新建并且更新flag
                    flag = DriveList.get(i).getBegin().getDayOfYear();
                    List<DrivingInformation> r = new ArrayList<DrivingInformation>();
                    r.add(DriveList.get(i));
                    result.add(r);
                }
            }

            return RespBean.ok("DrivingInformation are ",result);
        }else {
            return RespBean.error("is empty");
        }
    }


    /**
     * 上传驾驶中不良驾驶的图片
     * @param file
     * @param did
     * @return
     */
    @PostMapping("uploadDPicture")
    public RespBean uploadPicture(MultipartFile file, Integer did){
        Dpicture dpicture = new Dpicture();
        dpicture.setDid(did);
        String[] result = null;
        int num = dpictureMapper.selectCount(null);

        try {
            result = FastDFSUtil.upload(file.getBytes(),""+(num+1)+".jpg");
        } catch (IOException e) {
            e.printStackTrace();
        }
        dpicture.setUrl("http://47.102.99.215/"+result[0]+"/"+result[1]);
        dpictureMapper.insert(dpicture);
        return RespBean.ok("success",dpicture);
    }

    @GetMapping("getDPictureById")
    public RespBean getDPictureById(Integer did){
        QueryWrapper<Dpicture> queryWrapper = new QueryWrapper<Dpicture>();
        queryWrapper.eq("did",did);
        List<Dpicture>  dpictureList= dpictureMapper.selectList(queryWrapper);
        if(dpictureList==null){
            return RespBean.error("not have");
        }
        return RespBean.ok("success",dpictureList);
    }
}
