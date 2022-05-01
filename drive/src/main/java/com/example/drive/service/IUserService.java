package com.example.drive.service;

import com.example.drive.entity.User;
import com.baomidou.mybatisplus.extension.service.IService;
import com.example.drive.entity.UserHealth;
import org.springframework.stereotype.Service;

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

public interface IUserService extends IService<User> {
    void register(User u);
    User perfectInformation(User u);
    Long getUid();
    String getEmergencyNumber();
    User getUser();

}
