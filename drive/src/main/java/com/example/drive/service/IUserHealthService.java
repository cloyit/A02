package com.example.drive.service;

import com.example.drive.entity.DrivingInformation;
import com.example.drive.entity.UserHealth;
import com.baomidou.mybatisplus.extension.service.IService;

import java.time.LocalDateTime;
import java.util.List;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
public interface IUserHealthService extends IService<UserHealth> {
    List<UserHealth> getHealthByTime(LocalDateTime beginTime, LocalDateTime endTime);
    void uploadHealth(UserHealth userHealth);
    void uploadDriving(DrivingInformation drivingInformation);
}
