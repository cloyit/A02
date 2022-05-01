package com.example.drive.mapper;

import com.example.drive.entity.Brand;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.drive.entity.Title;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

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
public interface BrandMapper extends BaseMapper<Brand> {
    List<Brand> selectAllBrand();
    List<Brand> getBrandLimit(@Param("begin") int begin, @Param("size") int size);
}
