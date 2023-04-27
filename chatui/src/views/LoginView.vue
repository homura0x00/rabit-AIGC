<template>
  <div>
    <h1>Login Page</h1>
    <form @submit.prevent="login">
      <label>User Name</label>
      <input type="text" v-model="userName" required>
      <br>
      <label>Password</label>
      <input type="password" v-model="password" required>
      <br>
      <button type="submit">Log In</button>
    </form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      userName: '',
      password: '',
    }
  },
  methods: {
    login() {
      // var that = this;
      fetch('http:127.0.0.1:9000/api/login/',{
        method:'POST',
        headers: {
          'Conten-Type': 'application/json'
        },
        body:JSON.stringify({"username":this.userName,'password':this.password})
      }).then(response => {
        if(response.status == 200){
          return response.json()
        }else{
          return Promise.reject('登陆异常!')
        }
      }).then(
        function(r){
          console.log(r)
          if(r.status){
            localStorage.setItem('is_login','true')
            localStorage.setItem('token', r.token)
            localStorage.setItem('user_id', r.user_id)
            localStorage.setItem('user_name', r.user_name)
          }else{
            alert("登陆失败!")
          }
        }
      ).catch(error => alert(error))
    }
  }
}
</script>
