package com.example.drive.driveController;


import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.drive.entity.Brand;
import com.example.drive.entity.Car;
import com.example.drive.response.RespBean;
import com.example.drive.service.ICarService;
import com.example.drive.service.IUserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * <p>
 * 前端控制器
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
@RestController
@RequestMapping("/car")
public class CarController {
    @Autowired
    ICarService iCarService;
    @Autowired
    IUserService iUserService;

    /**
     * 根据当前用户增加车
     * @param car
     * @return
     */
    @PostMapping("addCar")
    public RespBean addCar(@RequestBody Car car) {
        //增加master_id
        Long uid = iUserService.getUid();
        car.setMasterId(uid);
        iCarService.addCar(car);
        return RespBean.ok("success and car is ",car);
    }

    /**
     * 根据token 获取所有的车
     * @return
     */
    @GetMapping("getAllCar")
    public RespBean getAllCar() {
        Long uid = iUserService.getUid();
        List<Car> cars = iCarService.getAllCarByUid(uid);
        return RespBean.ok("success and all cars are",cars);
    }

    /**
     * 根据car id 删除
     * @param Ids
     * @return
     */
    @PostMapping("deleteCarByIds")
    public RespBean deleteBrandById(@RequestBody List<Integer> Ids){
        //先判断有无数据
        if(!Ids.isEmpty()&&Ids.size()==0){
            return RespBean.error("empty");
        }
       iCarService.deleteCar(Ids);
        return RespBean.ok("Batch delete success");
    }



    /**
     * 绑定设备
     * @param
     * @return
     */
    @PostMapping("bandEquipment")
    public RespBean bandEquipment(@RequestBody Car c){
        QueryWrapper<Car> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("id",c.getId());
        iCarService.update(c,queryWrapper);
        return RespBean.ok("success",c);
    }
}
