package com.example.drive.controller;


import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.drive.entity.Detail;
import com.example.drive.entity.Title;
import com.example.drive.mapper.DetailMapper;
import com.example.drive.mapper.TitleMapper;
import com.example.drive.response.RespBean;
import com.example.drive.service.ITitleService;
import com.example.drive.service.impl.DetailServiceImpl;
import com.example.drive.utills.FastDFSUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.web.bind.annotation.*;

import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

/**

 *
 * @author zhulu
 * @since 2022-02-23
 */
@RestController
@RequestMapping("/detail")
public class DetailController {
    @Autowired
    DetailMapper detailMapper;
    @Autowired
    TitleMapper titleMapper;
    @Autowired
    ITitleService iTitleService;
    /**
     * 上传具体详情的接口
     * @param detail
     * @return
     */
    @PostMapping("uploadDetail")
    public RespBean uploadDetail(@RequestBody Detail detail){
        detailMapper.insert(detail);
        return RespBean.ok("success",detail);
    }

    /**
     * 为detail上传图片
     * @param file
     * @param id
     * @return
     */
    @PostMapping("uploadPictureById")
    public RespBean uploadPictureById(MultipartFile file,Integer id){
        //查询出对应的detail
        QueryWrapper<Detail> queryWrapper = new QueryWrapper<Detail>();
        queryWrapper.eq("detail_id",id);
        Detail detail = detailMapper.selectOne(queryWrapper);
        if(detail == null){
            return RespBean.error("id not exist");
        }
        String[] result = null;
        int num = detailMapper.selectCount(null);

        try {
            result = FastDFSUtil.upload(file.getBytes(),""+(num+1)+".jpg");
        } catch (IOException e) {
            e.printStackTrace();
        }
        detail.setPictureUrl("http://47.102.99.215/"+result[0]+"/"+result[1]);
        detailMapper.insert(detail);
        return RespBean.ok("success",detail);
    }


    /**
     * 根据titleId 查询所有的detail
     * @param titleId
     * @return
     */
    @GetMapping("getDetailByTitleId")
    public RespBean getDetailByTitleId(Integer titleId){
        //查询此titleId是不是还在
        Title title = iTitleService.getTitleById(titleId);
        if(title==null){
            return RespBean.error("The title corresponding to this ID does not exist");
        }else {
            //说明有titleid，查询即可
            QueryWrapper<Detail> queryWrapperDetail = new QueryWrapper<Detail>();
            queryWrapperDetail.eq("master_id",titleId);
            return RespBean.ok("success and detail are",detailMapper.selectList(queryWrapperDetail));
        }
    }


    /**
     * 根据id detail
     * @param
     * @return
     */
    @PostMapping("updateDetail")
    public RespBean updateDetail(@RequestBody Detail detail){


        QueryWrapper<Detail> queryWrapper = new QueryWrapper<Detail>();
        queryWrapper.eq("detail_id",detail.getDetailId());
        detailMapper.update(detail,queryWrapper);
        return RespBean.ok("success and detail is",detail);

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
        QueryWrapper<Detail> queryWrapper = new QueryWrapper<Detail>();
        queryWrapper.in("detail_id",Ids);
        detailMapper.delete(queryWrapper);
        return RespBean.ok("Batch delete success");
    }


}
