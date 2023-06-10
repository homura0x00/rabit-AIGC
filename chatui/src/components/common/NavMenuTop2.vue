<template>
    <!-- 插入一个导航栏 -->
    <el-menu
        router
        mode="horizontal"
        background-color="#252525"
        text-color="#fff"
        active-text-color="#ffd04b"
        style="width: 100%">

        <el-menu-item route="/" index="index">
            主页
        </el-menu-item>

        <!-- 中间的问答系统的标识 -->
        <span
            style=" background-color: #252525; position: absolute; color: #fff; padding-top: 20px; right: 43%; font-size: 20px; font-weight: bold"> Q & A system </span>

        <!-- 插入一个二级菜单栏，用户中心 -->
        <el-sub-menu index="person" style="position: absolute; right: 11px; padding-top: 10px">
            <template #title>
                <el-avatar :size="40" :src="circleUrl"></el-avatar>
            </template>
            <!-- 二级菜单里的组件 -->
            <el-menu-item route="/KGQA/personal" index="personal">
                个人信息
            </el-menu-item>
            <el-menu-item route="/" @click="logout()" index="logout">
                退出登录
            </el-menu-item>
        </el-sub-menu>


        <el-menu-item route="/KGQA/Chat" index="AnswerCard">
            医疗咨询
        </el-menu-item>
    </el-menu>
</template>

<script>
export default {
    name: "NavMenu2",
    methods: {
        logout() {
        },
    },
    data() {
        return {
            circleUrl: 'https//cube.elemencdn.com/3/7c/3ea6beec64369c2642b92c6726f1epng.png',
        }
    }
}
</script>

<style>
a {
    text-decoration: none;
}

span {
    pointer-events: none;
}
</style>