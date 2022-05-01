package com.example.drive.service;

import com.example.drive.entity.DrivingInformation;
import com.baomidou.mybatisplus.extension.service.IService;

import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
public interface IDrivingInformationService extends IService<DrivingInformation> {
    //查询一天的驾驶记录
    List<DrivingInformation> getDrivingInformationByDay(Long uid, LocalDateTime beginTime);
    //查询几天的
    List<DrivingInformation> getDrivingInformationByDay(Long uid, LocalDateTime beginTime,LocalDateTime endTime);

}
