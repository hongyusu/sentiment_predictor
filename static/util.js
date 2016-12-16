
function parse_datetime(str){
    var segs = str.split(" ");
    var dt = new Date(segs[0]);
    var hour = parseInt(segs[1].split(":")[0]);
    var minute = parseInt(segs[1].split(":")[1]);
    dt.setHours(hour);
    dt.setMinutes(minute);
    return dt;
}

function milisec2hour(t){
    // console.log(t);
    return t / 1000 / 3600;
}

function scaled_position_of_time(start, end, t, length){
    //time inputs should be Date object
    // console.log('end - start', end - start);
    // console.log('t, start', t, start);
    // console.log('t - start', t - start);
    var diff_hours_total = milisec2hour(end - start);
    var diff_hours_until_t = milisec2hour(t - start);
    // console.log(diff_hours_until_t, diff_hours_total);
    // console.log(length  * diff_hours_until_t / diff_hours_total);
    if(diff_hours_until_t == 0){
	return 0;
    }
    else{
	return length  * diff_hours_until_t / diff_hours_total;
    }
}

function time_anchors_between_period(start, end){
    //return 12:00am of all days between a period of time
    var cur_time = new Date(start);
    cur_time.setHours(12);
    cur_time.setMinutes(0);
    var ts = [];
    while(cur_time < end){
	ts.push(new Date(cur_time));
	cur_time.setDate(cur_time.getDate() + 1);
    }
    return ts;
}
