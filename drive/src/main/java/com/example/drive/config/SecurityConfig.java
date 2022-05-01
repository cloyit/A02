package com.example.drive.config;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.access.hierarchicalroles.RoleHierarchy;
import org.springframework.security.access.hierarchicalroles.RoleHierarchyImpl;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.builders.WebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

@Configuration
@EnableGlobalMethodSecurity(prePostEnabled = true)
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    /**
     * 配置Encoder注意一定要static 不然远端部署会报错
     * @return
     */
    @Bean
    public static BCryptPasswordEncoder getBCryptPasswordEncoder(){
        return new BCryptPasswordEncoder();
    }

    /**
     * 权限继承
     * @return
     */
    @Bean
    RoleHierarchy roleHierarchy(){
        RoleHierarchyImpl roleHierarchy = new RoleHierarchyImpl();
        String hierarchy = "ROLE_dba > ROLE_admin ROLE_admin > ROLE_stu";
        roleHierarchy.setHierarchy(hierarchy);
        return roleHierarchy;
    }

    @Autowired
    private PasswordEncoder passwordEncoder;
    @Autowired
    private UserDetailsService userDetailsService;

    /**
     * 配置用户，可以配置内存用户，也可以数据库
     * @param auth
     * @throws Exception
     */
    @Override
    protected void configure(AuthenticationManagerBuilder auth) throws Exception {
        auth.inMemoryAuthentication()
                .withUser("szw")
                .password(passwordEncoder.encode("whh")).roles("admin");//        super.configure(auth);
        auth.userDetailsService(userDetailsService)
                .passwordEncoder(passwordEncoder);
    }

    @Override
    public void configure(WebSecurity web) throws Exception {
        web.ignoring().antMatchers("*/js/**", "*/css/**","*/img/**");
        web.ignoring().antMatchers("/**/*.js", "/lang/*.json", "/**/*.css", "/**/*.js", "/**/*.map", "/**/*.html",
                "/**/*.png");
        web.ignoring().antMatchers(HttpMethod.GET,
                "/v2/api-docs",
                "/swagger-resources",
                "/swagger-resources/**",
                "/configuration/ui",
                "/configuration/security",
                "/swagger-ui.html/**",
                "/webjars/**",
                "/temp");
    }


    //在 protected void configure(HttpSecurity http)配置中添加下面这行代码：
    //http.cors().configurationSource(CorsConfigurationSource());

    //配置跨域访问资源
    private CorsConfigurationSource CorsConfigurationSource() {
        CorsConfigurationSource source =   new UrlBasedCorsConfigurationSource();
        CorsConfiguration corsConfiguration = new CorsConfiguration();
        corsConfiguration.addAllowedOrigin("*");	//同源配置，*表示任何请求都视为同源，若需指定ip和端口可以改为如“localhost：8080”，多个以“，”分隔；
        corsConfiguration.addAllowedHeader("*");//header，允许哪些header，本案中使用的是token，此处可将*替换为token；
        corsConfiguration.addAllowedMethod("*");	//允许的请求方法，PSOT、GET等
        ((UrlBasedCorsConfigurationSource) source).registerCorsConfiguration("/**",corsConfiguration); //配置允许跨域访问的url
        return source;
    }

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        //拦截设置
        http
                .authorizeRequests()
                .antMatchers("/auth/register").permitAll()
                .antMatchers("/testCreatWord").permitAll()
                .antMatchers("/user/getApp").permitAll()
                .antMatchers("/temp").permitAll()
                //在这里加条件来控制放行那个，加表达式，就是不知道该怎么判断是否登陆了，登录与没登录的区别在于
                //  SecurityContextHolder.getContext().setAuthentication(token);
                //它默认会去session找user来匹配权限，能不能让它去SecurityContextHolder.getContext().setAuthentication(token);里找
                //这里每一个请求我都在这里配置了，理论上可以通过注解+AOP在controller的方法上写注解
                .anyRequest().authenticated() //其他所有请求都需要认证
                //.antMatchers("/admin").hasAnyRole("user")  权限控制，配置角色
                //.antMatchers("/damin").hasAnyAuthority("ROLE_user")
                //所有的请求都得认证成功
                //.anyRequest().access("@rbacauthorityservice.hasPermission(request,authentication)") // RBAC 动态 url 认证
                //链式编程，and返回的就是http
                .and()
                .cors().configurationSource(CorsConfigurationSource())
                //跨域攻击不可用
                .and()
                //加入过滤器，后面的参数为把自己的过滤器放在哪里
                .addFilterBefore(new com.example.drive.config.JwtLoginFilter("/authentication/form",authenticationManager()), UsernamePasswordAuthenticationFilter.class)
                .addFilterBefore(new com.example.drive.config.JwtFilter(), UsernamePasswordAuthenticationFilter.class)
                .csrf().disable()
                //关掉session验证，开启JWT
                .sessionManagement().sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                .and().exceptionHandling();//匿名用户访问无权限资源时的异常处理;

    }
}
