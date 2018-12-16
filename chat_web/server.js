const express = require('express');
const bodyParser = require('body-parser');
var request = require('request');
const expressStatic = require('express-static');

//create server
var server = express();
server.listen(8083);

// cors
server.all('*', function(req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Content-Type,Content-Length, Authorization, Accept,X-Requested-With");
    res.header("Access-Control-Allow-Methods","PUT,POST,GET,DELETE,OPTIONS");
    res.header("X-Powered-By",' 3.2.1')
    if(req.method=="OPTIONS") res.send(200); // options请求快速返回 
    else next();
});

// body-parser json
server.use(bodyParser.json())
server.use(bodyParser.urlencoded({ extended: false }))

// router
server.use('/chat',(req, res)=>{
    console.log(req.body);
    request({
        url: 'http://openapi.tuling123.com/openapi/api/v2',
        method: "POST",
        json: true,
        headers: {
            "content-type": "application/json",
        },
        body: req.body
    }, function(error, response) {
        if (!error && response.statusCode == 200) {
            res.header("Content-Type", "application/json; charset=utf-8");
            res.end(JSON.stringify({ msg: response }));
        } else {
            res.status(500).end(JSON.stringify({err: 'server error'}))
        }
    });
});

// read file from ./www
server.use(expressStatic('./www'));