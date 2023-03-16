require('dotenv').config()

console.log("SMTP", process.env.SMTP_USER, process.env.SMTP_PASS)

const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const mongoose = require('mongoose')


const app = express();


app.use(express.json())
const multer = require('multer')
app.use(multer().any())
app.use(cookieParser())
app.use(cors())


app.use('/api', require('./routes/api/user-route'))

app.get('/', (req, res)=>{
    res.json({ page: 'home' });
})

app.use(require('./middlewares/error-middleware'))


const PORT = process.env.PORT || 3000;
async function start(){
    try{
        await mongoose.connect(process.env.DB_DNS, {
            useNewUrlParser: true,
        }).then(() => {
            console.log(`MongoDB connected`, `${process.env.DB_DNS}`);
        }).catch(err => {
            console.log('Failed to connect to MongoDB', err);
        });

        app.listen(PORT, ()=>{
            console.log(`server is listening on port ${PORT}`);
        })
    }catch(e){
        console.log(e)
    }
}
start();