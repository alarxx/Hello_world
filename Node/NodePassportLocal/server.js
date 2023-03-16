require('dotenv').config()

const express = require('express');

const app = express();

app.use(express.json())
const multer = require('multer')
app.use(multer().any())

app.get('/', (req, res)=>{
    res.json({ page: 'home' });
})

const PORT = process.env.PORT || 3000;
(async () => {
    try{
        app.listen(PORT, ()=>{
            console.log(`server is listening on port ${PORT}`);
        });
    }
    catch(e){
        console.log(e)
    }
})();