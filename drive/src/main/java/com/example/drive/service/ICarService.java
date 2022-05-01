package com.example.drive.service;

import com.example.drive.entity.Car;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
public interface ICarService extends IService<Car> {
     void addCar(Car car);
     List<Car> getAllCarByUid(Long uid);
     void deleteCar(List<Integer> ids);


}
