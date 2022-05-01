package com.example.drive.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.drive.entity.Car;
import com.example.drive.entity.UserCarRalation;
import com.example.drive.mapper.CarMapper;
import com.example.drive.mapper.UserCarRalationMapper;
import com.example.drive.service.ICarService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.drive.service.IUserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Collection;
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
public class CarServiceImpl extends ServiceImpl<CarMapper, Car> implements ICarService {
    @Autowired
    private IUserService iUserService;
    @Autowired
    private CarMapper carMapper;
    @Autowired
    private UserCarRalationMapper userCarRalationMapper;
    @Override
    @Transactional
    public void addCar(Car car) {
        UserCarRalation userCarRalation = new UserCarRalation();
        userCarRalation.setCarId(car.getId());
        userCarRalation.setUid(iUserService.getUid());
        userCarRalationMapper.insert(userCarRalation);
        carMapper.insert(car);
    }

    @Override
    public List<Car> getAllCarByUid(Long uid) {
        QueryWrapper<Car> queryWrapper = new QueryWrapper<Car>();
        queryWrapper.eq("master_id",uid);
        return carMapper.selectList(queryWrapper);
    }

    @Override
    public void deleteCar(List<Integer> ids) {
        QueryWrapper<Car> queryWrapper = new QueryWrapper<Car>();
        queryWrapper.in("id",ids);
        carMapper.delete(queryWrapper);
    }
}
