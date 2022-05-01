package com.example.drive.mapper;

import com.example.drive.entity.Brand;
import com.example.drive.entity.Title;
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
 * @since 2022-02-23
 */
@Mapper
public interface TitleMapper extends BaseMapper<Title> {
   List<Title> selectAllTitle();
   List<Title> getTitleLimit(@Param("begin") int begin, @Param("size") int size);
   Title getTitleById(@Param("id") int id);
   List<Title> getTitleLikeName(@Param("name") String name);
}
