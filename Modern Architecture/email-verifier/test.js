import http from 'k6/http';
import {check} from 'k6';

export const options = {
    vus: 10,
    duration: '30s'

};

export default function () {
    // code modified from the website you provided
    const url = 'http://localhost:8081/request-registration';
    const payload = JSON.stringify({"email": "test@example.com"});

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    let fails=0;
    const res=http.post(url, payload, params);
    check(res,{
        'status 204':(r)=>r.status===204
    })|| fails++;
}

export function handleSummary(data) {
    return {
        stdout: JSON.stringify(data, null, 2), // Print summary of results to stdout
    };
}