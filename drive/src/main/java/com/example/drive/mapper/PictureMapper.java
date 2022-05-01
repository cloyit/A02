package com.example.drive.mapper;

import com.example.drive.entity.Picture;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.web.bind.annotation.Mapping;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author zhulu
 * @since 2022-03-02
 */
@Mapper
public interface PictureMapper extends BaseMapper<Picture> {

}
