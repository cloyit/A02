package com.example.drive.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.drive.entity.DrivingInformation;
import com.example.drive.entity.UserHealth;
import com.example.drive.mapper.DrivingInformationMapper;
import com.example.drive.mapper.UserHealthMapper;
import com.example.drive.mapper.UserMapper;
import com.example.drive.service.IDrivingInformationService;
import com.example.drive.service.IUserHealthService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.drive.service.IUserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
@Service
public class UserHealthServiceImpl extends ServiceImpl<UserHealthMapper, UserHealth> implements IUserHealthService {

    @Autowired
    UserMapper userMapper;
    @Autowired
    UserHealthMapper userHealthMapper;
    @Autowired
    IUserService iUserService;
    @Autowired
    DrivingInformationMapper drivingInformationMapper;
    @Override
    public List<UserHealth> getHealthByTime(LocalDateTime beginTime, LocalDateTime endTime) {
            QueryWrapper<UserHealth> queryWrapper = new QueryWrapper<UserHealth>();
            queryWrapper.eq("uid",iUserService.getUid());
            queryWrapper.between("time",beginTime,endTime);
            return userHealthMapper.selectList(queryWrapper);
        }

    @Override
    public void uploadHealth(UserHealth userHealth) {

        userHealthMapper.insert(userHealth);
    }

    @Override
    public void uploadDriving(DrivingInformation drivingInformation) {
        drivingInformationMapper.insert(drivingInformation);
    }

}

