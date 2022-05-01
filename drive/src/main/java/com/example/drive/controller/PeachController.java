package com.example.drive.controller;


import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.example.drive.entity.Brand;
import com.example.drive.entity.Peach;
import com.example.drive.mapper.PeachMapper;
import com.example.drive.response.RespBean;
import com.example.drive.service.IPeachService;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.parameters.P;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zhulu
 * @since 2022-02-21
 */
@RestController
@RequestMapping("/peach")
public class PeachController {

    @Autowired
    PeachMapper peachMapper;
    @Autowired
    IPeachService iPeachService;
    //录入桃子
    @PostMapping("uploadPeach")
    public RespBean uploadPeach(@RequestBody Peach p){
        peachMapper.insert(p);
        return RespBean.ok("successs",p.getId());
    }

    /**
     * 根据id查询桃子
     * @param peachId
     * @return
     */
    @GetMapping("getPeach")
    public RespBean getPeach(@RequestParam("peachId") Integer peachId){
        QueryWrapper<Peach> queryWrapper = new QueryWrapper<Peach>();
        queryWrapper.eq("id",peachId);
        Peach p = peachMapper.selectOne(queryWrapper);
        p.setCnt(p.getCnt()+1);
        peachMapper.update(p,queryWrapper);
        return RespBean.ok("success",p);
    }

    /**
     * 查询所有的桃子
     * @return
     */
    @GetMapping("getAllPeach")
    public RespBean getAllPeach(){
        List<Peach> resultList = peachMapper.selectList(null);
        return RespBean.ok("success",resultList);

    }

    /**
     * 分页查询桃子
     * @param currentPage
     * @param size
     * @return
     */
    @GetMapping("getPeachLimit")
    public RespBean getPeachLimit(Integer currentPage,Integer size){

        //使用原生的sql即可
        //先查询所有的，然后再组装就好
        //先查询总数
        int begin = (currentPage-1)*size;
        int count = peachMapper.selectCount(null);
        List<Peach> peaches = null;
        peachMapper.getPeachLimit(begin,size);
        if((currentPage-1)*size>count){
            //说明没有那么多页数
            return RespBean.error("no much page are total is"+count);
        }else if(currentPage*size>count){
            //说明页数够但没有那么多数据
            //更新size
            size = count - (currentPage-1)*size;
        }
        peaches= peachMapper.getPeachLimit(begin,size);
        HashMap<String,Object> result = new HashMap<String,Object>();
        result.put("peaches",peaches);
        result.put("total",count);
       return RespBean.ok("success and peaches are total is"+count,result);
    }

    /**
     * 根据id 批量删除桃子
     * @param Ids
     * @return
     */
    @PostMapping("deleteBrandByIds")
    public RespBean deletePeachById(@RequestBody List<Integer> Ids){
        //先判断有无数据
        if(!Ids.isEmpty()&&Ids.size()==0){
            return RespBean.error("empty");
        }
        QueryWrapper<Peach> queryWrapper = new QueryWrapper<Peach>();
        queryWrapper.in("id",Ids);
        peachMapper.delete(queryWrapper);
        return RespBean.ok("Batch delete success");
    }

    /**
     * 更新桃子
     * @param peach
     * @return
     */
    @PostMapping("updatePeach")
    public RespBean updatePeach(@RequestBody Peach peach){
        QueryWrapper<Peach> queryWrapper = new QueryWrapper<Peach>();
        queryWrapper.eq("id",peach.getId());
        peachMapper.update(peach,queryWrapper);
        return RespBean.ok("update success and peach is",peach);
    }

    /**
     * 根据名称模糊查询桃子
     * @param LikeName
     * @return
     */
    @GetMapping("getPeachByLike")
    public RespBean getPeachByLike(String LikeName){
        QueryWrapper<Peach> queryWrapper = new QueryWrapper<Peach>();
        queryWrapper.like("name",LikeName);
        return RespBean.ok("success",peachMapper.selectList(queryWrapper));
    }


}
