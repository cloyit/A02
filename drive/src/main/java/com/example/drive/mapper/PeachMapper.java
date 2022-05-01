package com.example.drive.mapper;

import com.example.drive.entity.Peach;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author zhulu
 * @since 2022-02-21
 */
@Mapper
public interface PeachMapper extends BaseMapper<Peach> {

    List<Peach> getPeachLimit(@Param("begin") int begin,@Param("size") int size);
}
