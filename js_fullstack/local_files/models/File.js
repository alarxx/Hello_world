/**
 * Вряд ли нужен CRUD API для File
 * */
const mongoose = require('mongoose');
const {Schema} = require('mongoose');
const fs = require("fs");

require('./User'); // ref на User

const FileSchema = new Schema({
    path: {
        type: String,
        required: true
    },
    ownerId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        immutable: true,
        required: true
    },
    mimetype: {
        type: String,
        required: true
    },

    accessHoldersIds: [{ type: Schema.Types.ObjectId, ref: 'User' }],
    private: { type: Boolean, default: true },
});

FileSchema.plugin(require('mongoose-unique-validator'));

const BD_FILE_DIR = `data/files`;

/** Private. Законченный метод */
async function moveFile(multifile){
    try{
        const date = Date.now();

        const dir = `${BD_FILE_DIR}/${date}`;

        fs.mkdirSync(dir);

        const fullPath = `${dir}/${multifile.name}`

        await multifile.mv(fullPath);

        return fullPath;
    }catch(err){
        return null;
    }
}

/** Private. Законченный метод */
async function removeFile(path){
    try{
        fs.unlinkSync(path)
        fs.rmdirSync(require('path').join(path, '..'));
        return true;
    }catch(err){
        return false;
    }
}

/**
 * Метод сохраняет файл локально и сохраняет в базе mongo
 * function arguments: express-fileupload file and owner user id
 * returns {status, created_model, message}
 * */
FileSchema.statics.createAndMove = async function(multifile, userId){
    if(!userId)
        return {status: 'fail', message: 'createAndMove did not get userId'}

    // local storage
    const path = await moveFile(multifile)
    if(!path)
        return {status: 'fail', message: 'Can not move file'};

    // mongo
    try{
        const created = await new this({
            path: path,
            ownerId: userId,
            mimetype: multifile.mimetype,
        }).save();
        return {status: 'success', model: created};
    }
    catch(err){
        const errors = Object.keys(err.errors).map(key => err.errors[key].message);
        return {status: 'fail', message: errors};
    }
}

/**
 * delete by Id
 * returns deleted model
 * */
FileSchema.statics.deleteAndRemoveById = async function(id){
    const file = await this.findById(id);
    if(!file)
        return {status: 'fail', message: `Not found file with id ${id}`}

    return file.deleteAndRemove();
}

/**
 * Почему не воспользоваться сразу встроенным методом delete?
 * Метод нужен для удаления файла не только из mongo, а еще из локального хранилаща бинарного файла
 * if we already found model, we can call just delete
 * returns itself
 * */
FileSchema.methods.deleteAndRemove = async function(){
    // local storage
    const isRemoved = await removeFile(this.path)
    if(!isRemoved)
        return {status: 'fail', message: `Can not remove file on path ${this.path}`};

    // mongo
    try{
        await this.delete();
        return {status: 'success', model: this};
    }
    catch(err){
        return {status: 'fail', message: `Can not remove file from mongo ${this.id}}`};
    }
}

// Только сохранения и удаления будет достаточно пока
// FileSchema.methods.update = async function(file){}

const File = mongoose.model('File', FileSchema);

module.exports = File;
