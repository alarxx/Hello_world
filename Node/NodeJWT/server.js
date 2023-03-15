require('dotenv').config();

const express = require('express');

const app = express();

/** JSON Parsing */
// parse application/x-www-form-urlencoded
app.use(express.urlencoded({ extended: true }));
// parse application/json
app.use(express.json());

/** File Upload multipart/form-data */
app.use(require('./fileupload/multer-middleware')())

app.get('/', (req, res)=>{
    res.json({page: "home"});
})

app.post('/avatar', async (req, res)=>{

    const path = await req.files.avatar.save()

    res.json({...req.files.avatar, path });
})

app.use((err, req, res, next)=>{
    console.log("Catching error:", err);
    res.status(500).json({ error: err.message });
})

const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=>{
    console.log(`server is listening on port ${PORT}`);
})