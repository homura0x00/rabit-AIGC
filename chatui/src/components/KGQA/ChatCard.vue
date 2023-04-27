<template>
    <el-card class="box-card">
        <template #header>
            <span style="font-weight: bold; font-size: 15px;"> 医疗助理 </span>
        </template>
        <div id="dialog_container">
            <div v-for="oneDialog in text_dialog" :key="oneDialog">
                <el-divider content-position="left">{{ user_name }} -- {{ oneDialog.time }}</el-divider>
                <span id="question_card" style="font-size: 15px">{{ oneDialog.question }}</span>
                <el-divider content-position="right">回答</el-divider>
                <span id="answer_card">
                    <div style="font-size: 15px;" v-html="oneDialog.answer"></div>
                </span>
            </div>
        </div>
        <el-divider content-position="right"></el-divider>
        <el-input
            type="textarea"
            :autosize="{minRows: 2, maxRows: 4}"
            placeholder="请输入，与医疗卫生相关的话题，如：百日咳怎么治疗？"
            v-model="txt_question"
            >
        </el-input>
        <el-divider content-position="right">
            <el-button @click="ask_question()">Enter</el-button>
        </el-divider>
    </el-card>
</template>

<script>
import axios from 'axios';
import { ref } from 'vue';

export default {
    name: "AnswerCard",
    methods: {
        // 问答的框，每次提问完滚动条滚动到最下方新的消息
        scrollToBottom: function () {
            // 聊天记录框
            this.$nextTick(()=> {
                const div = document.getElementById('dialog_container')
                div.scrollTop = div.scrollHeight
            })

        },
        ask_question() {
            // 提问
            if (this.txt_question == '') {
                alert('输入不能为空！')
                return
            }
            // 添加一条问答对话
            const myDate = new Date();
            this.text_dialog.push({time: myDate.toLocaleString(), question: this.txt_question, answer: "我是一条测试答案"})
            this.scrollToBottom();
        }
    },
    data() {
        return {
            user_name: '默认用户',
            txt_question: '',
            text_dialog: [],
        }
    },
    setup() {
        const text = ref('')

        function getData() {
            axios({
                method:'get',
                url: 'https://127.0.0.1:9000/api'
            }).then((res)=>{
                console.log(res)
                text.value = res.data.text
            })
        }

        getData()

        return {
            text:text,
        }
    }
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
