package com.example.drive.mapper;

import com.example.drive.entity.Detail;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author zhulu
 * @since 2022-02-23
 */
@Mapper
public interface DetailMapper extends BaseMapper<Detail> {

    List<Detail> getDetailsById(Integer id);
    List<Detail> getDetailsByBId(Integer id);

}
