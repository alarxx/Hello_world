// await здесь не работает

(async ()=>{
    const fs = require('fs');
    const path = require('path');

    /*fs.stat('data', (err)=>{
        if(err){
            console.log('директории нет');
        } else {
            console.log('директория есть');
        }
    });
    */


    console.log('dir');
    fs.mkdirSync('data', (err)=>{
        if(err) console.log('not mkdir', err);
        else console.log('mkDir');  
    });
    console.log('mkDir');


    console.log("File!");

    await fs.writeFileSync('data/myText.txt',"Hello NodeJS",(err) => {
        if(err) console.log(err);
        else console.log("File has been saved!");
    });
    
    console.log("File has been saved!");


    /*console.log(path.join('data/myText2.txt', '..'));*/

    /*await fs.unlink('data/myText2.txt',(err) => {
        if(err) console.log(err);
        console.log('myText.txt was deleted');
    });*/

    /*await fs.rmdir('data', (err)=>{
        if(err)
            console.log('Директория не удалилась', err);
        else console.log('Директория удалилась');
    });*/
})();
