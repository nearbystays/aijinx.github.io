class Notifier {
    constructor() {
        this.i = 0;
        this.MAXITEMS = 10;
        this.settings_choices = [
            {name: 'msgonentry', type: 'choice', choice1: 'yes', choice2: 'no', defaultValue: 'yes', label: "Display Message 1 privately on entry - set to no for busy rooms"},
            {name:'msg1', type:'str', required: true, label:'Message 1',},
            {name:'msg2', type:'str', required: false, label:'Message 2',},
            {name:'msg3', type:'str', required: false, label:'Message 3',},
            {name:'msg4', type:'str', required: false, label:'Message 4',},
            {name:'msg5', type:'str', required: false, label:'Message 5',},
            {name:'msg6', type:'str', required: false, label:'Message 6',},
            {name:'msg7', type:'str', required: false, label:'Message 7',},
            {name:'msg8', type:'str', required: false, label:'Message 8',},
            {name:'msg9', type:'str', required: false, label:'Message 9',},
            {name:'msg10', type:'str', required: false, label:'Message 10',},
            {name:'msgcolor', type:'str', label:'Notice color (html code default dark red #9F000F)', defaultValue: '#9F000F'},
            {name: 'chat_ad', type:'int', minValue: 1, maxValue: 999, defaultValue: 2,
                label: 'Delay in minutes between notices being displayed (minimum 1)'}
        ];
    }

    onEnter(user) {
        if (cb.settings['msgonentry'] === 'yes') {
            cb.sendNotice('Welcome ' + user['user'] + '! ' + cb.settings['msg1'], user['user'], '', cb.settings['msgcolor'], 'bold');
        }
    }

    chatAd() {
        let msg;
        while (cb.settings['msg' + (this.i + 1)] === 0) {      //skip empty messages
            this.i++;
            this.i %= this.MAXITEMS;
        }
        msg = cb.settings['msg' + (this.i + 1)];
        this.i++;
        this.i %= this.MAXITEMS;
        cb.sendNotice(msg, '', '', cb.settings['msgcolor'], 'bold');
    }
}

let notifier = new Notifier();

cb.onEnter((user) => notifier.onEnter(user));