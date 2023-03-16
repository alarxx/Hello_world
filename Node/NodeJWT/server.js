require('dotenv').config()

const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const mongoose = require('mongoose')

const app = express();

app.use(express.json())
app.use(cookieParser())
app.use(cors())

app.get('/', (req, res)=>{
    res.json({ page: 'home' });
})

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