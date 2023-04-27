<template>
    知识图谱模块
</template>

<script>
export default {
    name: "KnowledgeGraph",
    
}
</script>

<style scoped>

.box-card {
    margin: 2% auto;
    width: 50%;
    min-width: 900px;
    text-align: left;
}

#dialog_container {
    overflow: auto;
    scroll-margin-right: 1px;
    /* 根据屏幕占用比设置高度 */
    min-height: calc(100vh - 360px);
    max-height: calc(100vh - 360px);
}
</style>