package com.example.drive.controller;


import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.drive.entity.Bmi;
import com.example.drive.entity.Brand;
import com.example.drive.entity.User;
import com.example.drive.mapper.BmiMapper;
import com.example.drive.response.RespBean;
import com.example.drive.service.IUserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zhulu
 * @since 2022-04-10
 */
@RestController
@RequestMapping("/bmi")
public class BmiController {


    @Autowired
    BmiMapper bmiMapper;
    @Autowired
    IUserService iUserService;
    /**
     * 上传bmi
     * @param bmi
     * @return
     */
    @PostMapping("uploadBmi")
    public RespBean updateBrandByName(@RequestBody Bmi bmi){

        return RespBean.error("weight is String not number");
    }

    /**
     * 获取用户所有的bmi历史数据
     * @return
     */
    @GetMapping("getAllBmi")
    public RespBean updateBrandByName(){
        User u = iUserService.getUser();
        QueryWrapper<Bmi> queryWrapper = new QueryWrapper<Bmi>();
        queryWrapper.eq("uid",u.getUid());
        List<Bmi> bmiList = bmiMapper.selectList(queryWrapper);
        return RespBean.ok("success and new brand is",bmiList);
    }
}
