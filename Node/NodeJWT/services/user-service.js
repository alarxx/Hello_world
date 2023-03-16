const UserModel = require('../models/user-model')
const bcrypt = require('bcrypt')
const uuid = require('uuid')

const mailService = require('./mail-service');
const tokenService = require('./token-service');
const UserDto = require('../dtos/user-dto')
const ApiError = require('../exceptions/api-error')

class UserService {

    async registration(email, password){
        const candidate = await UserModel.findOne({ email });

        if(candidate){
            throw ApiError.BadRequest(`User ${email} already exists`);
        }
        const hashPassword = await bcrypt.hash(password, 3);
        const activationLink = uuid.v4();

        const user = await UserModel.create({
            email,
            password: hashPassword,
            activationLink,
        });

        await mailService.sendActivationMail(email, `${process.env.API_URL}/api/activate/${activationLink}`);

        const userDto = new UserDto(user);
        const userDtoObject = {...userDto}
        const tokens = await tokenService.generateTokens(userDtoObject);

        await tokenService.saveToken(userDtoObject.id, tokens.refreshToken);

        return {...tokens, user: userDtoObject};
    }

    async activate(activationLink){
        const user = await UserModel.findOne({ activationLink });
        if(!user){
            throw ApiError.BadRequest('Некорректная ссылка активации');
        }
        user.isActivated = true;
        return await user.save();
    }

}

module.exports = new UserService();