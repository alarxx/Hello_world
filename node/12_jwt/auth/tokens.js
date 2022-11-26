const {credentials} = require(`../config`);
const visibleUserData = require('./visibleUserData');

const jwt = require('jsonwebtoken');

async function generate(payload){
  const JWT_ACCESS_TOKEN = await jwt.sign(payload, credentials.JWT_ACCESS_TOKEN, { expiresIn: '15s' });
  const JWT_REFRESH_TOKEN = await jwt.sign(payload, credentials.JWT_REFRESH_TOKEN, { expiresIn: '30s' });
  return {JWT_ACCESS_TOKEN, JWT_REFRESH_TOKEN};
}

const RefreshTokenModel = require('../models/RefreshToken');

// Активирует, создает или обновляет токен
async function createOrUpdate(token, user){
  const user_id = user.id;

  const model = await RefreshTokenModel.findOne({user_id});

  if(!model){ //create
    console.log('create');
    RefreshTokenModel.create({
      user_id,
      token
    });
  }
  else { //update
    console.log('update');
    RefreshTokenModel.findByIdAndUpdate(model.id, {
      token,
      active: true
    });
  }
}

// if active update
async function updateIfActive(token, user){
  const model = await RefreshTokenModel.findOne({user_id: user.id});
  if(!model){
    throw new Error('jwt not found');
  }
  else if(model.active){
    console.log('updated');
    RefreshTokenModel.findByIdAndUpdate(model.id, {token});
  }
  else{
    throw new Error('jwt is deactivated');
  }
}

// Находит и деактивирует
async function deactivate(user){
  const model = await RefreshTokenModel.findOne({user_id: user.id});

  if(!model){
    throw new Error('No such user');
  }
  else{
    RefreshTokenModel.findByIdAndUpdate(model.id, {active: false});
  }
}


// const decoded = await jwt.verify(access_token, credentials.JWT_ACCESS_TOKEN);
// Создает на основе req.user после LocalStrategy
module.exports.create = () => async (req, res)=>{
  if(req.signedCookies.JWT_REFRESH_TOKEN)
    return refresh();

  try{
    const userData = visibleUserData(req.user);

    const {JWT_ACCESS_TOKEN, JWT_REFRESH_TOKEN} = await generate(userData);

    await createOrUpdate(JWT_REFRESH_TOKEN, userData);

    res.cookie('JWT_REFRESH_TOKEN', JWT_REFRESH_TOKEN, {signed: true});

    res.json({
      status: 'success',
      type: 'create',
      ...userData,
      JWT_ACCESS_TOKEN
    });
  }
  catch(err){
    res.json({
      status: 'fail',
      type: 'create',
      message: err.message,
    });
  }
}

// Обновляет на основе signedCookies.JWT_REFRESH_TOKEN
module.exports.refresh = () => async (req, res)=>{
  try{
    const decoded = await jwt.verify(req.signedCookies.JWT_REFRESH_TOKEN, credentials.JWT_REFRESH_TOKEN);

    const userData = visibleUserData(decoded);
    const {JWT_ACCESS_TOKEN, JWT_REFRESH_TOKEN} = await generate(userData);

    await updateIfActive(JWT_REFRESH_TOKEN, userData);

    res.cookie('JWT_REFRESH_TOKEN', JWT_REFRESH_TOKEN, {signed: true});

    res.json({
      status: 'success',
      type: 'refresh',
      ...userData,
      JWT_ACCESS_TOKEN
    });
  }
  catch(err){
    res.json({
      status: 'fail',
      type: 'refresh',
      message: err.message,
    });
  }
}

module.exports.delete = () => async (req, res)=>{
  try{
    const decoded = await jwt.verify(req.signedCookies.JWT_REFRESH_TOKEN, credentials.JWT_REFRESH_TOKEN);

    const userData = visibleUserData(decoded);
    // active = false
    await deactivate(userData);

    res.clearCookie('JWT_REFRESH_TOKEN');

    res.json({
      status: 'success',
      type: 'delete',
    });
  }
  catch(err){
    res.json({
      status: 'fail',
      type: 'delete',
      message: err.message,
    });
  }
};
