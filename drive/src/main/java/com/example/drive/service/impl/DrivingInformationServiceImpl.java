package com.example.drive.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.conditions.query.QueryChainWrapper;
import com.example.drive.entity.DrivingInformation;
import com.example.drive.mapper.DrivingInformationMapper;
import com.example.drive.service.IDrivingInformationService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
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
public class DrivingInformationServiceImpl extends ServiceImpl<DrivingInformationMapper, DrivingInformation> implements IDrivingInformationService {

    @Autowired
    DrivingInformationMapper drivingInformationMapper;

    /**
     * 根据日期获取驾驶记录，精确到日就好
     * @param beginTime
     * @return
     */
    @Override
    public List<DrivingInformation> getDrivingInformationByDay(Long uid, LocalDateTime beginTime) {

        LocalDateTime endTime = LocalDateTime.of(beginTime.getYear(),beginTime.getMonth(),beginTime.getDayOfMonth()+1,0,0);
        QueryWrapper<DrivingInformation> queryWrapper = new QueryWrapper<DrivingInformation>();
        queryWrapper.eq("uid",uid);
        queryWrapper.between("begin",beginTime,endTime);
        return drivingInformationMapper.selectList(queryWrapper);
    }

    @Override
    public List<DrivingInformation> getDrivingInformationByDay(Long uid, LocalDateTime beginTime, LocalDateTime endTime) {
         QueryWrapper<DrivingInformation> queryWrapper = new QueryWrapper<DrivingInformation>();
        queryWrapper.eq("uid",uid);
        queryWrapper.between("begin",beginTime,endTime);
        return drivingInformationMapper.selectList(queryWrapper);
    }
}
