const mongoose = require('mongoose');
const uniqueValidator = require('mongoose-unique-validator'); //for errors

const UserSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
    },
    email: {
        type: String,
        unique: true,
        lowercase: true,
        minLength: 11,
        required: true,
    },
    password: {
        type: String,
        required: true,
    },

    roles: {
        type: String,
        enum: ['client', 'manager', 'admin'],
        lowercase: true,
        default: 'client',
    },
});

UserSchema.plugin(uniqueValidator);

const UserModel = mongoose.model('User', UserSchema);

module.exports = UserModel;
