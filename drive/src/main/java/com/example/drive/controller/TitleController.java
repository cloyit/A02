package com.example.drive.controller;


import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.drive.entity.Brand;
import com.example.drive.entity.Detail;
import com.example.drive.entity.Peach;
import com.example.drive.entity.Title;
import com.example.drive.mapper.DetailMapper;
import com.example.drive.mapper.TitleMapper;
import com.example.drive.response.RespBean;
import com.example.drive.service.IDetailService;
import com.example.drive.service.ITitleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * <p>
 *  前端控制器
 * </p>
 * * title 的controller
 *  * 对于表与实体类不同的情况，应该使用service
 * @author zhulu
 * @since 2022-02-23
 */
@RestController
@RequestMapping("/title")
public class TitleController {



    @Autowired
    private TitleMapper titleMapper;
    @Autowired
    private DetailMapper detailMapper;

    /**
     *
     *  1 获取所有的title, 根据每一个title 获取detail
     * @return
     */
    @PostMapping("getAllTitle")
    public RespBean getAllTitle(){

        List<Title> titles = titleMapper.selectAllTitle();
        //查出所有的detail id 组合成title


        for(Title title : titles){
            QueryWrapper<Detail> queryWrapper = new QueryWrapper<Detail>();
            queryWrapper.eq("master_id",title.getTitleId());
            List<Detail> details=  detailMapper.selectList(queryWrapper);
            title.setDetails(details);
        }
        return RespBean.ok("title is",titles);
    }

    /**
     * 上传title
     * @param title
     * @return
     */
    @PostMapping("uploadTitle")
    public RespBean uploadTitle(@RequestBody Title title){
        titleMapper.insert(title);
        return RespBean.ok("success",title);

    }



    /**
     * 根据id 批量删除title
     * @param Ids
     * @return
     */
    @PostMapping("deleteTitleByIds")
    public RespBean deleteTitleByIds(@RequestBody List<Integer> Ids){
        //先判断有无数据
        if(!Ids.isEmpty()&&Ids.size()==0){
            return RespBean.error("empty");
        }
        QueryWrapper<Title> queryWrapper = new QueryWrapper<Title>();
        queryWrapper.in("title_id",Ids);
        titleMapper.delete(queryWrapper);
        return RespBean.ok("Batch delete success");
    }


    /**
     * 根据id 更新title
     * @param
     * @return
     */
    @PostMapping("updateTitle")
    public RespBean updateTitle(@RequestBody Title title){


        QueryWrapper<Title> queryWrapper = new QueryWrapper<Title>();
        queryWrapper.eq("title_id",title.getTitleId());
        titleMapper.update(title,queryWrapper);
        return RespBean.ok("success and titile is",title);

    }

    /**
     * 根据名称模糊查询title
     * @param LikeName
     * @return
     */
    @GetMapping("getTitleByLike")
    public RespBean getTitleByLike(String LikeName){

        return RespBean.ok("success",titleMapper.getTitleLikeName(LikeName));
    }
    /**
     * 分页查询title
     * @param currentPage
     * @param size
     * @return
     */
    @GetMapping("getTitleLimit")
    public RespBean getTitleLimit(Integer currentPage,Integer size){
        //title 的分页查询有点特殊，必须自己组装detail
        //使用原生的sql即可
        //先查询所有的，然后再组装就好
        //先查询总数
        int begin = (currentPage-1)*size;
        int count = titleMapper.selectCount(null);
        List<Title> titleList = null;
        titleMapper.getTitleLimit(begin,size);
        if((currentPage-1)*size>count){
            //说明没有那么多页数
            return RespBean.error("no much page are total is"+count);
        }else if(currentPage*size>count){
            //说明页数够但没有那么多数据
            //更新size
            size = count - (currentPage-1)*size;
        }
        titleList= titleMapper.getTitleLimit(begin,size);
        //根据title的id 查询detail

        for(Title title : titleList){
            QueryWrapper<Detail> queryWrapper = new QueryWrapper<Detail>();
            queryWrapper.eq("master_id",title.getTitleId());
            List<Detail> details=  detailMapper.selectList(queryWrapper);
            title.setDetails(details);
        }
        HashMap<String,Object> result = new HashMap<String,Object>();
        result.put("peaches",titleList);
        result.put("total",count);
        return RespBean.ok("success and peaches are total is"+count,result);
    }



}
