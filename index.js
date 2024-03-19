var APP = {
    NAME: "Spin To Win",
    NOTICE: 'Spin To Win',
    DATE: new Date().getFullYear() + "-" + (new Date().getMonth() + 1) + "-" + new Date().getDate(),
    DEV: 'Jinx',
};

var WHEEL = {
    label: cb.settings.adTxt || 'Spin To Win Is ACTIVE!',
    enable: true,
    size: 40,
    timer: timeLord(cb.settings.wheelAdTimer, "Spin To Win"),
    timerID: "",
    adTxt: "",
    adTxt2: "",
    adTxtShort: "",
    rewards: [],
    rarePrizes: [],
    rareLength: 0,
    winName: [],
    winPrize: [],
    winRare: [],
    spinCost: cb.settings.cost_per_spin,
    spinCostBackUp: 0,
    discount: 0,
    sale: false,
    saleTimer: timeLord(cb.settings.wheelAdTimer / 2, "Spin To Win"),
    saleTimerID: "",
    spinExact: cb.settings.spin_exact === "Yes",
    maxMulti: "",
    multiExact: cb.settings.multispin_exact === "Yes",
    noSpinOver: 0,
    noSpinList: [],
    noSpinOn: false,
    rareOn: false,
    rareChance: 1 / cb.settings.rareChance,
    rareMin: cb.settings.rareMin,
    rareBonus: cb.settings.rareBonus === "Yes",
    modSpin: cb.settings.modSpin === "Yes",
    background: "",
    foreground: "",
    rollCount: 0,
    msgReward: "",
};

function settingInit() {
    let setChoiceAppInfo = [{
        name: 'Hi001',
        label: 'Hi ' + cb.room_slug + '. for questions or bug reports direct message us at @t836874.',
        required: !1,
        type: 'choice'
    }, {
        name: 'Hi002',
        label: ' ------------------------------------------------------------------------------------------------- ' +
            ' ------------------------------>   ' + APP.NAME.toUpperCase() + '  <---------------------------------- ' +
            ' ----------------------------------->   ' + APP.DATE + '  <----------------------------------------- ' +
            ' ---------------------------------------------------------------------------------------------------- ',
        required: !1,
        type: 'choice'
    }, {
        name: 'Hi003',
        label: '>>> RECENT UPDATE: You can now run a sale. Type "/wheelsale help" for more info. <<<',
        required: !1,
        type: 'choice'

    }];

    let setChoiceWheel = [{
            //name: 'wheelh',
            //label: '------------------------------>  WHEEL OF FORTUNE  <-------------------- ',
            //required: !1,
            //type: 'choice'
            //}, {
            //name: 'use_wheel',
            //label: 'Would you like to use the Spin To Win?',
            //type: 'choice',
            //choice1: 'Yes',
            //choice2: 'No',
            //defaultValue: 'No'
            //},
            //{
            name: 'cost_per_spin',
            type: 'int',
            minValue: 1,
            maxValue: 999,
            defaultValue: 25,
            label: 'Cost Per Spin Of The Wheel (1-999): '
        },
        {
            name: 'wheelAdTimer',
            type: 'int',
            minValue: 2,
            maxValue: 999,
            defaultValue: 5,
            label: 'How Often To Advertise The Bot: (min)'
        },
        {
            name: 'adTxt',
            type: 'str',
            minLength: 1,
            maxLength: 255,
            label: 'Custom text to promote the wheel:',
            defaultValue: 'Spin To Win Is ACTIVE!',
            required: false
        },
        {
            name: 'wheel_txtcolor',
            type: "str",
            minLength: 6,
            maxLength: 7,
            defaultValue: '#FFFFFF',
            label: 'Pick the color of the font ( default: White #FFFFFF):',
            required: false
        },
        {
            name: "wheel_bgcolor",
            type: "str",
            minLength: 6,
            maxLength: 7,
            label: "Background color :",
            defaultValue: "#011f3f",
            required: false
        },
        {
            name: "modSpin",
            type: "choice",
            choice1: 'Yes',
            choice2: 'No',
            defaultValue: 'Yes',
            label: "Do you want to allow mods to use the Free spin command when needed?",
        },
        {
            name: 'spin_exact',
            type: 'choice',
            choice1: 'Yes',
            choice2: 'No',
            defaultValue: 'No',
            label: 'Only spin if the exact token price set for wheel is tipped? (Will disable all the bonus spin parameter)'
        },
        {
            name: 'multispin_exact',
            type: 'choice',
            choice1: 'Yes',
            choice2: 'No',
            defaultValue: 'No',
            label: 'Only spin if the tip is an exact multiple of cost? (Up to the maximum number of bonus spin)'
        },
        {
            name: 'multispin_count',
            type: 'int',
            minValue: 0,
            maxValue: 99,
            defaultValue: 4,
            label: 'How Many Bonus Spins To Allow Per Tip? (0-99)'
        },
        {
            name: 'nospin_over',
            type: 'int',
            minValue: 0,
            maxValue: 999,
            defaultValue: 0,
            label: 'Prevent the wheel from spinning if the tip amount is over X ( To turn off that feature, leave it at 0)'
        },
        {
            name: 'nospin',
            type: 'str',
            minLength: 1,
            maxLength: 255,
            label: 'Exclude specific tip amounts from spinning the wheel. Separate each amount by a comma ( ex: 150, 200, 300).',
            required: false
        },
        {
            name: 'rarePrize',
            type: 'str',
            minLength: 1,
            maxLength: 255,
            label: 'If you want a prize to be rare, enter it here. You can add more than one, separeted by ";".',
            required: false
        },
        {
            name: 'rareChance',
            type: 'int',
            minValue: 1,
            maxValue: 500,
            defaultValue: 99,
            label: ' Chance of rare prize - 1 in X? (the higher the number, less chance of getting the prize, up to 500)',
            required: false
        },
        {
            name: 'rareMin',
            type: 'int',
            minValue: 2,
            maxValue: 120,
            defaultValue: 21,
            label: ' Rare will only happen after X roll',
            required: false
        },
        {
            name: 'rareBonus',
            type: 'choice',
            choice1: 'Yes',
            choice2: 'No',
            defaultValue: 'No',
            label: 'Should the rare prize only happen on bonus spins to encourage big tippers?'
        }
    ];

    var setChoiceWheelRewards = [];

    for (let i = 1; i <= WHEEL.size; i++) {
        setChoiceWheelRewards.push({
            name: 'pos' + i,
            label: 'Reward #' + i,
            type: 'str',
            required: false
        });
    }
    return setChoiceAppInfo.concat(setChoiceWheel, setChoiceWheelRewards);
}

cb.settings_choices = settingInit();
//********************** End of settings **********************
//********************** Start of Messages **********************

function sendDev(message) {
    cb.sendNotice(APP.NAME + " to Dev: " + message, APP.Dev, "#A020F0", "#FFFFFF", 'bold');
}

function sendError(username, message) {
    cb.sendNotice(APP.NAME + " Error: " + message, username, "#FFFFFF", "#FF0000", 'bold');
}

function sendSuccessMessage(username, message) {
    cb.sendNotice(APP.NAME + " to " + (username === cb.room_slug ? "You" : username) + ": " + message, username, "#468847", "#FFFFFF", 'bold');
}


function sendNotMod(username) {
    cb.sendNotice('**** Only mods and broadcasters can use this command. ****', username, "#FFFFFF", "#FF0000");
}

//********************** Start of utilities **********************
function timeLord(time, iden) {
    let timer = parseFloat(time);
    if (timer < 1) {
        sendError(cb.room_slug, "Time lapse for " + iden + " is to short. Using default value.");
        timer = 3;
    }
    timer *= 30000;
    timer = parseInt(timer);
    return timer;
}

function colorChecker(c, d, s) {
    //c: color to check, d: default color, s: settings.
    if (c) {
        if ((c.length === 3 || c.length === 6) && /^[0-9A-F]+$/i.test(c)) {
            return "#" + c;
        } else if ((c.length === 4 || c.length === 7) && /^#[0-9A-F]+$/i.test(c)) {
            return c;
        } else {
            cb.sendNotice(APP.NAME + " - Error while setting " + s + ". It has to be in a HEX format. Using default value: " + d + ".", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
            return (d);
        }
    } else {
        cb.sendNotice('\u25BA ' + s + ': Not set.', cb.room_slug, '', '#db0', 'bold');
        return (d);
    }
}

function formatArray(arr, andor) {
    let outStr = "";
    if (arr.length === 1) {
        outStr = arr[0];
    } else if (arr.length === 2) {
        //joins all with "and" but no commas
        //example: "bob and sam"
        outStr = arr.join(' ' + andor + ' ');
    } else if (arr.length > 2) {
        //joins all with commas, but last one gets ", and" (oxford comma!)
        //example: "bob, joe, and sam"
        outStr = arr.slice(0, -1).join(', ') + ', ' + andor + ' ' + arr.slice(-1);
    }
    return outStr;
}

function parseUsername(username) {
    if (username) {
        username = username.replace(/[^a-zA-Z0-9 _]+/g, "");
        return username.toLowerCase();
    }
    return;
}

function removeWordsAtBeginning(message, number) {
    // Split the message into words
    const words = message.trim().split(/\s+/);

    // Remove the specified number of words from the beginning
    const remainingWords = words.slice(number);

    // Join the remaining words back into a string and trim any trailing spaces
    const trimmedMessage = remainingWords.join(' ').trim();

    return trimmedMessage;
}

//********************** End of utilities **********************

function spinWheel(spinCount, u) {
    let randomnumber;
    cb.sendNotice('**** ' + u + ' is spinning The Wheel! ****', "", WHEEL.bgColor, WHEEL.txtColor, "bold");

    if (WHEEL.rareOn && !WHEEL.rareBonus && WHEEL.rollCount >= WHEEL.rareMin && Math.random() < WHEEL.rareChance) {
        if (WHEEL.rareLength > 1) {
            randomnumber = Math.floor(Math.random() * (WHEEL.rareLength));
        } else {
            randomnumber = 0;
        }
        cb.sendNotice('**** The Wheel Stops On : ' + WHEEL.rarePrizes[randomnumber] + '!!!! Lucky you!!!', "", WHEEL.bgColor, WHEEL.txtColor, "bold");
        WHEEL.winName.push(u);
        WHEEL.winPrize.push(WHEEL.rarePrizes[randomnumber]);
        WHEEL.winRare.push(true);
        WHEEL.rollCount++;
    } else {
        randomnumber = Math.floor(Math.random() * (WHEEL.rewardsLength));
        cb.sendNotice('**** The Wheel Stops On : ' + WHEEL.rewards[randomnumber], "", WHEEL.bgColor, WHEEL.txtColor, "bold");
        WHEEL.winName.push(u);
        WHEEL.winPrize.push(WHEEL.rewards[randomnumber]);
        WHEEL.winRare.push(false);
        WHEEL.rollCount++;
    }
    if (spinCount > 1) {
        cb.sendNotice('*** ! Multi-Spin Bonus Activated ! ***', "", WHEEL.bgColor, WHEEL.txtColor, "bold");
        for (let x = 1; x < spinCount; x++) {
            if (WHEEL.rareLength > 0 && WHEEL.rollCount >= WHEEL.rareMin && Math.random() < WHEEL.rareChance) {
                if (WHEEL.rareLength > 1) {
                    randomnumber = Math.floor(Math.random() * (WHEEL.rareLength));
                } else {
                    randomnumber = 0;
                }
                cb.sendNotice('**** Bonus Spin Stops On : ' + WHEEL.rarePrizes[randomnumber] + '!!!! Lucky you!!!', "", WHEEL.bgColor, WHEEL.txtColor, "bold");
                WHEEL.winName.push(u);
                WHEEL.winPrize.push(WHEEL.rarePrizes[randomnumber]);
                WHEEL.winRare.push(true);
                WHEEL.rollCount++;
            } else {
                randomnumber = Math.floor(Math.random() * (WHEEL.rewardsLength));
                cb.sendNotice('**** Bonus Spin Stops On : ' + WHEEL.rewards[randomnumber], "", WHEEL.bgColor, WHEEL.txtColor, "bold");
                WHEEL.winName.push(u);
                WHEEL.winPrize.push(WHEEL.rewards[randomnumber]);
                WHEEL.winRare.push(false);
                WHEEL.rollCount++;
            }
        }
    }
    cb.sendNotice('**********************************', "", WHEEL.bgColor, WHEEL.txtColor);
}

function wheelAd() {
    if (WHEEL.token > 4) {
        cb.sendNotice(WHEEL.adTxt + WHEEL.adTxt2, "", WHEEL.bgColor, WHEEL.txtColor, 'bold');
        WHEEL.token = 1;
    } else {
        cb.sendNotice(WHEEL.adTxtShort, "", WHEEL.bgColor, WHEEL.txtColor, 'bold');
        WHEEL.token += 1;
    }
    WHEEL.timerID = cb.setTimeout(wheelAd, (WHEEL.timer));
}

function wheelAdCleanUp(adTxt) {

    //Fixing the ad text.
    if (adTxt == "") {
        adTxt = 'Spin To Win Is ACTIVE!';
    }
    WHEEL.adTxtShort = adTxt + '\n Type "/wheelinfo" for more information.';
    WHEEL.adTxt = adTxt;
}

function wheelAdSecondLine() {
    let adTxt;
    adTxt += WHEEL.adTxt2;
    adTxt += '\n Type "/rewards" to get the list of prizes';
    return adTxt;
}

function spamCleanUp() {
    WHEEL.adTxtShort = 'Tip  ' + (WHEEL.spinCost) + ' tokens to spin the wheel.';
    WHEEL.adTxt2 = "";
    if (WHEEL.sale) {
        WHEEL.adTxt2 += '\n The cost of spinning the wheel is currently on sale!';
    }
    if (WHEEL.spinExact) {
        WHEEL.adTxt2 += '\n The wheel will only spin for an exact tip of ' + (WHEEL.spinCost) + ' tokens.';
    } else {
        if (WHEEL.multiExact) {
            if (WHEEL.maxMulti > 6) {
                WHEEL.adTxt2 += '\n The wheel will only spin for multiples of ' + (WHEEL.spinCost) + ' tokens up to ' + WHEEL.maxMulti * WHEEL.spinCost + " tokens.";
            } else if (WHEEL.maxMulti > 0) {
                let tipsList = [];
                for (let x = 1; x <= WHEEL.maxMulti + 1; x++) {
                    let tipsListTemp = x * WHEEL.spinCost;
                    if (!cbjs.arrayContains(WHEEL.noSpinList, tipsListTemp)) {
                        tipsList.push(tipsListTemp);
                    }
                }
                WHEEL.adTxt2 += '\n The wheel will only spin for tips of ' + formatArray(tipsList, 'or') + " tokens.";
            } else {
                WHEEL.adTxt2 += '\n The wheel will only spin for multiples of ' + (WHEEL.spinCost) + ' tokens.';
            }
        } else {
            WHEEL.adTxt2 += '\n Tip  ' + (WHEEL.spinCost) + ' tokens to spin the wheel.';
            if (WHEEL.maxMulti > 0) {
                WHEEL.adTxt2 += '\n Bonus spins are enabled. Tip at least ' + (WHEEL.spinCost * 2) + ' tokens to get a bonus spin.' +
                    "\n You can gain " + (WHEEL.maxMulti > 1 ? " up to " : "") + WHEEL.maxMulti + " bonus spin" + (WHEEL.maxMulti > 1 ? "s" : "") + " per tip.";
            }
            if (WHEEL.noSpinOn) {
                WHEEL.adTxt2 += '\n The wheel will not spin for these values: ' + WHEEL.noSpinList.join(', ') + (WHEEL.noSpinOver > 0 ? " and everything over " + WHEEL.noSpinOver : "") + ".";
            } else if (WHEEL.noSpinOver > 0) {
                WHEEL.adTxt2 += '\n The wheel will not spin for tips over ' + WHEEL.noSpinOver + '.';
            }
            if (WHEEL.rareOn && WHEEL.rareBonus && !WHEEL.spinExact) {
                WHEEL.adTxt2 += '\n To have a chance at winning the rare prize, you have to trigger the bonus spin by tipping ' + (WHEEL.spinCost * 2) + ' or more tokens';
            }
        }
    }
}

function changePrice() {
    excludeList();
    logicCheck();
    spamCleanUp();
    wheelAdCleanUp(WHEEL.label);
}

function wheelStartAd() {
    let msgInit =
        '--------------------------------' +
        '\n Spin To Win by 4science' +
        '\n Type "/wheelhelp" to see all the commands.' +
        '\n --------------------------------';

    msgInit += WHEEL.adTxt2;
    msgInit += '\n Type /rewards to see all the prizes.' +
        '\n --------------------------------';
    cb.sendNotice(msgInit, "", WHEEL.bgColor, WHEEL.txtColor, 'bold');
}

function msgRewards() {
    WHEEL.rewardsLength = WHEEL.rewards.length;
    //Lets generate the prize list.
    WHEEL.msgReward = '**** Here is the list of prizes on the wheel ****\n';
    for (let x = 0; x < WHEEL.rewardsLength; x++) {
        WHEEL.msgReward += String('Reward ' + (x + 1) + ': ' + WHEEL.rewards[x] + '\n');
    }
    if (WHEEL.rareLength > 0) {
        WHEEL.msgReward += (WHEEL.rareBonus ? "Trigger the bonus spins " : "Spin the wheel ") + "to get a chance of winning " +
            (WHEEL.rareLength > 1 ? "one of the rare prizes" : "the rare prize") + "\n";

        for (let x = 0; x < WHEEL.rareLength; x++) {
            WHEEL.msgReward += String('Rare reward ' + (x + 1) + ': ' + WHEEL.rarePrizes[x] + '\n');
        }
    }
    WHEEL.msgReward += '---------------------------------\n';
    WHEEL.msgReward += '      Good Luck And Have Fun.    \n';
    WHEEL.msgReward += '---------------------------------';
}

function rareSetup() {
    if (cb.settings.rarePrize != "" || WHEEL.rareLength > 0) {
        if (WHEEL.spinExact && WHEEL.rareBonus) {
            cb.sendNotice(' Wheel - Error - Rare prizes are set to only happen on bonus spin and the bot is set to only spin on the exact cost.\n Rare prizes are disabled.', cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
            WHEEL.rareOn = false;
        } else if (WHEEL.rareBonus && WHEEL.maxMulti === 0 || WHEEL.rareBonus && WHEEL.spinExact) {
            cb.sendNotice(' Wheel - Error - Rare is set to only happen on bonus spin and bonus spin is disabled.\n Rare prizes are disabled.', cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
            WHEEL.rareOn = false;
        } else {
            //Splitting the rare prizes
            let rareTemp = cb.settings.rarePrize;
            if (rareTemp) {
                let rareTemp2 = rareTemp.split(";");
                for (let i = 0; i < rareTemp2.length; i++) {
                    WHEEL.rarePrizes.push(' :mrstar ' + rareTemp2[i] + ' :mrstar ');
                }
            }
            WHEEL.rareLength = WHEEL.rarePrizes.length;

            if (WHEEL.rareLength > 0) {
                WHEEL.rareOn = true;
            } else {
                cb.sendNotice("Wheel - Error while setting the rare prizes.", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
            }
        }
    }
}

function excludeList() {
    WHEEL.noSpinList = [];
    WHEEL.noSpinOver = cb.settings.nospin_over;
    if (cb.settings.nospin && !WHEEL.spinExact) {
        let tempExcl = cb.settings.nospin.split(/[,\s]+/);
        for (let i = 0; i < tempExcl.length; i++) {
            let tempExcl2 = parseInt(tempExcl[i]);
            if (tempExcl[i] !== "" && tempExcl[i] !== 0) {
                if (tempExcl[i].charAt(0) == ">") {
                    let tempNoSpin = parseInt(tempExcl[i].substr(1));
                    if (isNaN(tempNoSpin) || tempNoSpin < 2) {
                        cb.sendNotice("Wheel - Error while processing the \"No Spin Over\" amount.", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
                    } else if (tempNoSpin < WHEEL.spinCost) {
                        cb.sendNotice("Wheel - Error while processing the \"No Spin Over\" amount. It appears lower than the spin cost.", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
                    } else if (isNaN(WHEEL.noSpinOver)) {
                        WHEEL.noSpinOver = tempNoSpin;
                    } else if (WHEEL.noSpinOver === 0) {
                        WHEEL.noSpinOver = tempNoSpin;
                    } else {
                        cb.sendNotice("Wheel - Error - You have indicated two different values for do not spin over X amount. Picking the smallest one.", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
                        if (WHEEL.noSpinOver > tempNoSpin) {
                            WHEEL.noSpinOver = tempNoSpin;
                        }
                    }
                } else if (tempExcl2 === WHEEL.spinCost) {
                    cb.sendNotice("Wheel - Error - You can not exlude the value of a spin.", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
                } else if (!isNaN(tempExcl2) && tempExcl2 > 0) {
                    WHEEL.noSpinList.push(tempExcl2);
                } else {
                    cb.sendNotice("Wheel - Error while processing the value " + tempExcl[i] + " from the exclude list.", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
                }
            }
        }
        if (WHEEL.noSpinList.length !== 0) {
            WHEEL.noSpinList.sort(function(a, b) {
                return b - a;
            });
            for (let i = 0; i <= WHEEL.noSpinList.length; i++) {
                if (0 < WHEEL.noSpinOver && WHEEL.noSpinList[i] >= WHEEL.noSpinOver || WHEEL.noSpinList[i] < WHEEL.spinCost) {
                    if (WHEEL.noSpinList[i] == (WHEEL.noSpinOver)) {
                        WHEEL.noSpinOver--;
                    }
                    WHEEL.noSpinList.splice(i, 1);
                    i--;
                }
            }
            WHEEL.noSpinOn = true;
            WHEEL.noSpinList.sort(function(a, b) {
                return a - b;
            });
        }
        if (WHEEL.noSpinList.length === 0) {
            WHEEL.noSpinOn = false;
        }
    }
}

function logicCheck() {
    WHEEL.maxMulti = cb.settings.multispin_count;
    if (WHEEL.noSpinOver > 0 && !WHEEL.spinExact) {
        if (WHEEL.noSpinOver == WHEEL.spinCost && !WHEEL.spinExact) {
            WHEEL.spinExact = true;
        } else if (WHEEL.noSpinOver < WHEEL.spinCost) {
            cb.sendNotice('Wheel - Error while setting up the "don\'t spin over" value. It appears lower than the spin cost. Disabling.', cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
            WHEEL.noSpinOver = 0;
        } else if (WHEEL.spinCost * (WHEEL.maxMulti + 1) > WHEEL.noSpinOver) {
            WHEEL.maxMulti = Math.floor(WHEEL.noSpinOver / WHEEL.spinCost) - 1;
            cb.sendNotice('Wheel - Error - It is impossible to do that many bonus spin for under ' + WHEEL.noSpinOver + ' tokens.\n Bonus spin has been set to ' + WHEEL.maxMulti, cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
        }
    }
    if (WHEEL.spinExact) {
        if (WHEEL.maxMulti > 0) {
            cb.sendNotice('Wheel - Error - The wheel is set to only spin on an exact tip of ' + WHEEL.spinCost + ", the wheel will disable the bonus spin. Make sure this is what you want.", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
            WHEEL.maxMulti = 0;
        }
    }
}

function saleNotifier() {
    cb.sendNotice("The cost of spinning the wheel is " + WHEEL.discount + "% off!\nTip  " + (WHEEL.spinCost) + ' tokens to spin the wheel.', "", WHEEL.bgColor, WHEEL.txtColor, 'bold');
    WHEEL.saleTimerID = cb.setTimeout(saleNotifier, WHEEL.saleTimer);
}

function initWheel() {
    //Lets check the color
    WHEEL.txtColor = colorChecker(cb.settings.wheel_txtcolor, '#FFFFFF', "Wheel text color");
    WHEEL.bgColor = colorChecker(cb.settings.wheel_bgcolor, '#0B7762', "Wheel background color");

    //Removing empties.
    for (let i = 1; i <= WHEEL.size; i++) {
        if (cb.settings['pos' + i]) {
            if (cb.settings['pos' + i].charAt(0) === '-') {
                cb.sendNotice("Wheel to Broadcaster - Prize " + i + ": " + cb.settings['pos' + i].substr(1) + ', has been disabled for today. \nIf this is an error, please restart the bot and remove the "-" in front of that prize.', cb.room_slug, "#FFFFFF", "#FF0000", "bold");
            } else {
                WHEEL.rewards.push(cb.settings['pos' + i]);
            }
        }
    }

    //Lets exclude some numbers.
    excludeList();

    //Logic Check/re
    logicCheck();

    spamCleanUp();

    //Taking care of the rare prizes.
    rareSetup();

    // Clean up and prepare the advertisement text
        wheelAdCleanUp(WHEEL.label);

    //Advert on start
    wheelStartAd();

    //Reward list
    msgRewards();

    //Start the ad
    cb.setTimeout(wheelAd, WHEEL.timer);
}

cb.onTip(function(tip) {
    let spinCount = 0;
    if (WHEEL.spinExact) {
        if (tip.amount === WHEEL.spinCost) {
            spinCount = 1;
        }
    } else if (WHEEL.noSpinOver === 0 || WHEEL.noSpinOver > 0 && tip.amount <= WHEEL.noSpinOver) {
        if (!WHEEL.noSpinOn || WHEEL.noSpinOn && !cbjs.arrayContains(WHEEL.noSpinList, tip.amount)) {
            if (!WHEEL.multiExact || WHEEL.multiExact && tip.amount % WHEEL.spinCost === 0 && tip.amount <= WHEEL.spinCost * (1 + WHEEL.maxMulti)) {
                spinCount = Math.min(Math.floor(tip.amount / WHEEL.spinCost), WHEEL.maxMulti + 1);
            }
        }
    }
    if (spinCount > 0) {
        let u = tip.from_user;
        if (tip.is_anon_tip) {
            u = "An anonymous user";
        }
        spinWheel(spinCount, u);
    }
});

cb.onMessage(function(m) {
    if (m.m.charAt(0) === "/") {
        let silentCmd = m['X-Spam'] === true;
        let u = m.user;
        let message = m.m.split(" ");
        let isMod = (cb.room_slug === u || m.is_mod || u === '4science');
        let noticeMsg;
        m['X-Spam'] = true;
        m.background = '#d9d9d9';
        switch (message[0]) {
            case "/rewards":
            case "/srewards":
            case "/prizes":
                if (isMod && message['0'] != '/srewards') {
                    u = '';
                }
                cb.sendNotice(WHEEL.msgReward, u, WHEEL.bgColor, WHEEL.txtColor, 'bold');
                return m;
            case "/wheelinfo":
            case "/swheelinfo":
                if (isMod && message[0] != '/swheelinfo') {
                    u = '';
                }
                cb.sendNotice(WHEEL.adTxt + WHEEL.adTxt2, u, WHEEL.bgColor, WHEEL.txtColor, 'bold');
                return m;
            case "/winners":
            case "/swinners":
            case "/winner": {
                if (isMod && message['0'] != '/swinners') {
                    u = '';
                }
                let cmdVar1 = message[1];
                let cmdVar2 = parseInt(message[1]);
                let nameNum;
                let l = WHEEL.winName.length;
                if (l === 0) {
                    cb.sendNotice('**** No one has won a prize yet ****', u, WHEEL.bgColor, WHEEL.txtColor);
                    return m;
                }
                if (isNaN(cmdVar1) && cmdVar1 !== undefined) {
                    const target = parseUsername(cmdVar1);
                    if (cmdVar1 === "all" || cmdVar1 === "All") {
                        if (l > 100) {
                            noticeMsg = '**** Here is the list of the last 100 winners! ****\n';
                            nameNum = l - 100;
                        } else {
                            noticeMsg = '**** Here is the list of all the winners! ****\n';
                            nameNum = 0;
                        }
                        for (let x = nameNum; x < l; x++) {
                            noticeMsg += 'Roll #' + (x + 1) + ': ' + WHEEL.winName[x] + ' won ' + WHEEL.winPrize[x] + '\n';
                        }
                        noticeMsg += '**************************************';
                        cb.sendNotice(noticeMsg, u, WHEEL.bgColor, WHEEL.txtColor);
                        return m;
                    } else if (cbjs.arrayContains(WHEEL.winName, target)) {
                        noticeMsg = '**** Here is the list of all the prizes ' + target + ' won! ****\n';
                        let prizesWon = [];
                        for (let x = 0; x < l; x++) {
                            if (target === WHEEL.winName[x]) {
                                prizesWon.push(WHEEL.winPrize[x]);
                            }
                        }
                        noticeMsg += prizesWon.join(', ');
                        noticeMsg += '\n**************************************';
                        cb.sendNotice(noticeMsg, u, WHEEL.bgColor, WHEEL.txtColor);
                    } else {
                        cb.sendNotice('Unable to find ' + target + ' on the winners list.', u, WHEEL.bgColor, WHEEL.txtColor);
                    }
                } else {
            if (cmdVar1 === undefined || cmdVar1 === '') {
                        cmdVar2 = 10;
                    }
                    if (cmdVar2 <= 0) {
                        cb.sendNotice('**** Here are the last 0 winners!  :p', u, WHEEL.bgColor, WHEEL.txtColor);
                        return m;
                    }
                    if (l < cmdVar2) {
                        cmdVar2 = l;
                    }
                    noticeMsg = '**** Here are the last ' + cmdVar2 + ' winners! **** \n';
                    for (let x = l - cmdVar2; x < l; x++) {
                        noticeMsg += 'Roll #' + (x + 1) + ': ' + WHEEL.winName[x] + ' won ' + WHEEL.winPrize[x] + '\n';
                    }
                    noticeMsg += '**************************************';
                    cb.sendNotice(noticeMsg, u, WHEEL.bgColor, WHEEL.txtColor);
                }
                return m;
            }
            case "/whowon":
            case "/swhowon": {
                if (isMod && message['0'] !== '/swhowon') {
                    u = '';
                }
                let winners = [];
                let l = WHEEL.winName.length;
                if (message[1] === 'rare') {
                    for (let x = 0; x < l; x++) {
                        if (WHEEL.winRare[x] === true) {
                            winners.push(WHEEL.winName[x] + " won " + WHEEL.winPrize[x]);
                        }
                    }
                    if (winners.length > 0) {
                        noticeMsg = '**** Here is everyone who won a rare prize! **** \n';
                        noticeMsg += winners.join('\n');
                        noticeMsg += '\n**************************************';
                    } else {
                        noticeMsg = '**** No one has won a rare prize yet. **** ';
                    }
                } else {
                    let cmdVar1 = parseInt(message[1]) - 1;
                    if (cmdVar1 >= 0) {
                        for (let x = 0; x < l; x++) {
                            if (WHEEL.winPrize[x] === WHEEL.rewards[cmdVar1]) {
                                winners.push(WHEEL.winName[x]);
                            }
                        }
                        if (winners.length > 0) {
                            noticeMsg = '**** Here is everyone who won ' + WHEEL.rewards[cmdVar1] + '! **** \n';
                            noticeMsg += winners.join(', ');
                            noticeMsg += '\n**************************************';
                        } else {
                            noticeMsg = '**** No one has won ' + WHEEL.rewards[cmdVar1] + ' yet. **** ';
                        }
                    } else {
                        noticeMsg = 'You have to indicate the position of the reward. Use /rewards to see the list. You can use "/whowon rare" for the rare prize.';
                    }
                }
                cb.sendNotice(noticeMsg, u, WHEEL.bgColor, WHEEL.txtColor);
                return m;
            }
            case "/addprize": {
                if (!isMod) {
                    cb.sendNotice("Only mods and broadcasters can use this command.", u, "#FFFFFF", "#FF0000");
                } else {
                    let label;
                    if (!message[1]) {
                        cb.sendNotice("You need to include a label for that option.", u, "#FFFFFF", "#FF0000");
                        return m;
                    }
                    for (let j = 1; j < message.length; j++) {
                        if (j === 1) {
                            label = message[j];
                        } else {
                            label += " " + message[j];
                        }
                    }
                    cb.sendNotice("Wheel to Broadcaster - " + (u === cb.room_slug ? "You" : u) + ' added "' + label + '" to the list of prizes.', cb.room_slug, "#FFFFFF", "#FF0000", "bold");
                    cb.sendNotice("Wheel to mods - " + u + ' added "' + label + '" to the list of prizes.', "", "#FFFFFF", "#FF0000", "bold", "red");
                    WHEEL.rewards.push(label);
                    msgRewards();
                }
                return m;
            }
            case "/removeprize":
            case "/deleteprize":
            case "/delprize": {
                if (!isMod) {
                    cb.sendNotice("Only mods and broadcasters can use this command.", u, "#FFFFFF", "#FF0000");
                } else {
                    if (WHEEL.rewardsLength == 0) {
                        cb.sendNotice("The list of prizes is already empty.", u, "#FFFFFF", "#FF0000");
                        return m;
                    }
                    let position = parseInt(message[1]) - 1;
                    if (!position && position != 0) {
                        cb.sendNotice("You need to indicate the position of the prize you want to remove. Type /rewards to see each position.", u, "#FFFFFF", "#FF0000");
                        return m;
                    }
                    if (position >= WHEEL.rewardsLength) {
                        cb.sendNotice("It seems like there isn't that many prizes on the wheel. Type /rewards to see each position.", u, "#FFFFFF", "#FF0000");
                        return m;
                    }
                    cb.sendNotice("Wheel to Broadcaster - " + (u === cb.room_slug ? "You" : u) + ' removed "' + WHEEL.rewards[position] + '" from the list of prizes.', cb.room_slug, "#FFFFFF", "#FF0000", "bold");
                    cb.sendNotice("Wheel to mods - " + u + ' removed "' + WHEEL.rewards[position] + '" from the list of prizes.', "", "#FFFFFF", "#FF0000", "bold", "red");
                    WHEEL.rewards.splice(position, 1);
                    msgRewards();
                }
                return m;
            }
            case "/addrare": {
                if (!isMod) {
                    cb.sendNotice("Only mods and broadcasters can use this command.", u, "#FFFFFF", "#FF0000");
                } else {
                    let label;
                    if (!message[1]) {
                        cb.sendNotice("You need to include a label for that option.", u, "#FFFFFF", "#FF0000");
                        return m;
                    }
                    for (let j = 1; j < message.length; j++) {
                        if (j === 1) {
                            label = message[j];
                        } else {
                            label += " " + message[j];
                        }
                    }
                    cb.sendNotice("Wheel to Broadcaster - " + (u === cb.room_slug ? "You" : u) + ' added "' + label + '" to the list of rare prizes.', cb.room_slug, "#FFFFFF", "#FF0000", "bold");
                    cb.sendNotice("Wheel to mods - " + u + ' added "' + label + '" to the list of rare prizes.', "", "#FFFFFF", "#FF0000", "bold", "red");
                    WHEEL.rarePrizes.push(' :mrstar ' + label + ' :mrstar ');
                    WHEEL.rareLength = WHEEL.rarePrizes.length;

                    if (WHEEL.rareLength > 0) {
                        rareSetup();
                    } else {
                        cb.sendNotice("Wheel - Error while setting the rare prizes.", cb.room_slug, "#FFFFFF", "#FF0000", 'bold');
                    }
                    msgRewards();
                }
                return m;
            }
            case "/removerare":
            case "/deleterare":
            case "/delrare": {
                if (!isMod) {
                    cb.sendNotice("Only mods and broadcasters can use this command.", u, "#FFFFFF", "#FF0000");
                } else {
                    if (WHEEL.rareLength == 0) {
                        cb.sendNotice("The list of rare prizes is already empty.", u, "#FFFFFF", "#FF0000");
                        return m;
                    }
                    let position = parseInt(message[1]) - 1;
                    if (!position && position != 0) {
                        cb.sendNotice("You need to indicate the position of the rare prize you want to remove. Type /rewards to see each position.", u, "#FFFFFF", "#FF0000");
                        return m;
                    }
                    if (position >= WHEEL.rareLength) {
                        cb.sendNotice("It seems like there isn't that many rare prizes on the wheel. Type /rewards to see each position.", u, "#FFFFFF", "#FF0000");
                        return m;
                    }
                    cb.sendNotice("Wheel to Broadcaster - " + (u === cb.room_slug ? "You" : u) + ' removed "' + WHEEL.rarePrizes[position] + '" from the list of rare prizes.', cb.room_slug, "#FFFFFF", "#FF0000", "bold");
                    cb.sendNotice("Wheel to mods - " + u + ' removed "' + WHEEL.rarePrizes[position] + '" from the list of rare prizes.', "", "#FFFFFF", "#FF0000", "bold", "red");
                    WHEEL.rarePrizes.splice(position, 1);
                    WHEEL.rareLength = WHEEL.rarePrizes.length;
                    if (WHEEL.rareLength == 0) {
                        WHEEL.rareOn = false;
                    }
                    msgRewards();
                }
                return m;
            }
            case "/wheelexcluded":
                if (WHEEL.noSpinOn) {
                    cb.sendNotice('The wheel will not spin for these values: ' + WHEEL.noSpinList.join(', ') + (WHEEL.noSpinOver > 0 ? " and everything over " + WHEEL.noSpinOver + "." : "."), u, "#FFFFFF", "#FF0000", "bold");
                } else if (WHEEL.noSpinOver > 0) {
                    cb.sendNotice("The wheel will not spin for values over " + WHEEL.noSpinOver + ".", u, "#FFFFFF", "#FF0000", "bold");
                }
                return m;
            case "/freespin":
                if (isMod) {
                    if (WHEEL.modSpin || u === cb.room_slug) {
                        if (u !== cb.room_slug) {
                            cb.sendNotice(u + ' is using the free spin function.', cb.room_slug, "#FFFFFF", "#FF0000", "bold");
                        }
                        if (message[1] !== undefined && message[1] !== '') {
                            u = parseUsername(message[1]);
                        }
                        spinWheel(1, u);
                    } else {
                        cb.sendNotice('**** This command is not enable for mods ****', u, "#FFFFFF", "#FF0000");
                    }
                } else {
                    cb.sendNotice('**** Only mods and broadcasters can use this command. ****', u, "#FFFFFF", "#FF0000");
                }

                return m;
            case "/wheeldesc":
            case "/wheeltext":
                if (!isMod) {
                    sendNotMod(u);
                    return m;
                }

                const label = removeWordsAtBeginning(m.m, 1);
                if (!label) {
                    cb.sendNotice("You need to include a label for that option.", u, "#FFFFFF", "#FF0000");
                    return m;
                }
                WHEEL.label = label;
                wheelAdCleanUp(WHEEL.label);
                sendSuccessMessage(u, 'The wheel ad is now ' + label);
                return m;
            case "/spincost":
            case "/wheelprice":
                if (isMod) {
                    let cmdVar1 = parseInt(message[1]);
                    if (cmdVar1 > 0) {
                        WHEEL.spinCost = cmdVar1;
                        if (u !== cb.room_slug) {
                            cb.sendNotice(u + ' changed the price to spin the wheel.', cb.room_slug, "#FFFFFF", "#FF0000", "bold");
                        }
                        //assume that it close the sale if it was on.
                        if (WHEEL.sale) {
                            cb.cancelTimeout(WHEEL.saleTimerID);
                            WHEEL.spinCostBackUp = WHEEL.spinCost;
                            WHEEL.sale = false;
                            cb.sendNotice("Using /spincost during a sale turns the sale message off. Use /wheelsale to change the price and keep the sale messge on.", u, "#FFFFFF", "#FF0000");
                        }
                        if (WHEEL.noSpinOn) {
                            cb.sendNotice("You have a list of excluded numbers. Using /spincost may overwrite some of those restrictions.", u, "#FFFFFF", "#FF0000", "bold");
                        }
                        cb.sendNotice('**** The price to spin the wheel is now ' + WHEEL.spinCost + " token" + (WHEEL.spinCost > 1 ? "s" : "") + "!", "", WHEEL.bgColor, WHEEL.txtColor, "bold");
                        changePrice();
                    } else {
                        cb.sendNotice('**** You have to indicate a new price. ex: "/spincost 30" ', u, WHEEL.bgColor, WHEEL.txtColor);
                    }
                } else {
                    cb.sendNotice('**** Only mods and broadcasters can use this command. ****', u, "#FFFFFF", "#FF0000");
                }
                return m;
            case "/wheelsale": {
                WHEEL.discount = parseInt(message['1']);
                if (!message['1'] || message['1'] === "help" || isNaN(WHEEL.discount) && !(message['1'] === "off" || WHEEL.discount === 0)) {
                    cb.sendNotice('/wheelsale help \n You can use a percentage like "/wheelsale 25%" to reduce the price by 25%. \n You can also use it without the percentage to have a specific price like "/wheelsale 50" to set the sale price at 50 tokens.\n You can turn the sale off by typing /"wheelsale off".', u, "#FFFFFF", "#35454f", 'bold');
                } else if (!isMod) {
                    cb.sendNotice('**** Only mods and broadcasters can use this command. ****', u, "#FFFFFF", "#FF0000");
                } else {
                    if (message['1'] === "off" || WHEEL.discount === 0) {
                        if (!WHEEL.sale) {
                            cb.sendNotice("The cost of spinning the wheel isn't on sale right now.", "", "#FFFFFF", "#FF0000");
                        } else {
                            cb.cancelTimeout(WHEEL.saleTimerID);
                            WHEEL.spinCost = WHEEL.spinCostBackUp;
                            cb.sendNotice("The sale is now off. \n You now have to tip " + WHEEL.spinCost + " tokens to spin the wheel.", "", WHEEL.bgColor, WHEEL.txtColor, "bold");
                            WHEEL.sale = false;
                            changePrice();
                        }
                    } else if (message['1'].endsWith("%")) {
                        if (WHEEL.discount < 0 || WHEEL.discount >= 100) {
                            cb.sendNotice("X has be be a number between 0 and 99. It will be the percentage taken off the price.", u, "#FFFFFF", "#FF0000");
                        } else {
                            if (WHEEL.sale) {
                                WHEEL.spinCost = WHEEL.spinCostBackUp;
                            } else {
                                WHEEL.saleTimerID = cb.setTimeout(saleNotifier, WHEEL.saleTimer);
                                WHEEL.spinCostBackUp = WHEEL.spinCost;
                            }
                            WHEEL.spinCost = Math.ceil(WHEEL.spinCost * (1 - WHEEL.discount / 100));
                            if (WHEEL.noSpinOn) {
                                cb.sendNotice("You have a list of excluded numbers. Using /wheelsale may overwrite some of those restrictions.", u, "#FFFFFF", "#FF0000", "bold");
                            }
                            cb.sendNotice("The cost of spinning the wheel is now " + WHEEL.discount + "% off!\nTip " + (WHEEL.spinCost) + ' tokens to spin the wheel.', "", WHEEL.bgColor, WHEEL.txtColor, 'bold');
                                                    WHEEL.sale = true;
                            changePrice();
                        }
                    } else if (WHEEL.discount > 0) {
                        if (!WHEEL.sale && WHEEL.discount > WHEEL.spinCost || WHEEL.sale && WHEEL.discount > WHEEL.spinCostBackUp) {
                            cb.sendNotice("To be a sale, the new price needs to be under the cost of a spin.", u, "#FFFFFF", "#FF0000");
                        } else {
                            if (!WHEEL.sale) {
                                WHEEL.saleTimerID = cb.setTimeout(saleNotifier, WHEEL.saleTimer);
                                WHEEL.spinCostBackUp = WHEEL.spinCost;
                            }
                            WHEEL.spinCost = WHEEL.discount;
                            WHEEL.discount = 100 - Math.round(WHEEL.discount / WHEEL.spinCostBackUp * 100);
                            if (WHEEL.noSpinOn) {
                                cb.sendNotice("You have a list of excluded numbers. Using /wheelsale may overwrite some of those restrictions.", u, "#FFFFFF", "#FF0000", "bold");
                            }
                            cb.sendNotice("The cost of spinning the wheel is now " + WHEEL.discount + "% off!\mTip  " + (WHEEL.spinCost) + ' tokens to spin the wheel.', "", WHEEL.bgColor, WHEEL.txtColor, 'bold');
                                                    WHEEL.sale = true;
                            changePrice();
                        }
                    }
                }
                return m;
            }
            case "/wheeltxtcolor":
                if (isMod) {
                    WHEEL.txtColor = colorChecker(message[1], '#FFFFFF', "Wheel text color 1");
                } else {
                    cb.sendNotice('**** Only mods and broadcasters can use this command. ****', u, "#FFFFFF", "#FF0000");
                }
                return m;
            case "/wheelbgcolor":
                if (isMod) {
                    WHEEL.bgColor = colorChecker(message[1], '#0B7762', "Wheel text color 1");
                } else {
                    cb.sendNotice('**** Only mods and broadcasters can use this command. ****', u, "#FFFFFF", "#FF0000");
                }
                return m;
            case "/wheelhelp":
                cb.sendNotice("Here are the commands available for the Spin To Win.\n" +
                    "- /rewards: Show all the prizes in chat.\n" +
                    "- /wheelinfo: See how much to tip to spin the wheel.\n" +
                    "- /wheeltext Y: Change the advertisement for the wheel.\n" +
                    "- /winners: Show the last 10 winners.\n" +
                    "- /winners X: Show the last X winners.\n" +
                    "- /winners all: Show all the requests (up to 100).\n" +
                    "- /whowon X: Show everyone who won the prize at position X.\n" +
                    "- /whowon rare: Show everyone who won a rare prize.\n" +
                    "- /freespin - Spin the wheel if you are out of ideas or if you just want to reward someone. Will only work if you are the broadcaster, can be turn on for mods.\n" +
                    "- /freespin namehere - Will spin the wheel but will use the namehere as the username. Useful to have the right name in the winning list at the end.\n" +
                    '- /addprize Y: Add a prize name Y to the wheel.\n' +
                    "- /removeprize X: Removes the prize at position X.\n" +
                    "- /addrare Y: Add a rare prize name Y to the wheel.\n" +
                    "- /removerare X: Remove the rare prize at position X.\n" +
                    "- /wheelsale X: Change the price of spinning to X. The bot will promote it as a sale.\n" +
                    "- /wheelsale X%: Reduce the price to spin the wheel by X%. The bot will promote it as a sale.\n" +
                    "- /wheelsale help: More info on how to use /wheelsale.\n" +
                    "- /wheelsale off: Ends the sale. Returns the spin cost to original price\n" +
                    "- /spincost X: Change the price to spin the wheel to X, without promoting it as a sale.", u, '', '#35454f', "bold");
                return m;
            default:
                if (!silentCmd) {
                    m['X-Spam'] = false;
                    m.background = '';
                }
                return m;
        }
    }
    return m;
});

cb.onEnter(function(user) {
    let u = user.user;
    let msgOnEnter = "--------------------------------";
    msgOnEnter += "\n Hello " + u + ", " + WHEEL.adTxt;

    msgOnEnter += WHEEL.adTxt2;
    msgOnEnter += '\n Type /rewards to see all the prizes.';
    msgOnEnter += "\n --------------------------------";
    cb.sendNotice(msgOnEnter, u, WHEEL.bgColor, WHEEL.txtColor, 'bold');
});

initWheel()const sqlite3 = require('sqlite3').verbose();

class Database {
    constructor(dbFilePath) {
        this.db = new sqlite3.Database(dbFilePath, (err) => {
            if (err) {
                console.log('Could not connect to database', err);
            } else {
                console.log('Connected to database');
            }
        });
    }

    run(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.run(sql, params, function (err) {
                if (err) {
                    console.log('Error running sql ' + sql);
                    console.log(err);
                    reject(err);
                } else {
                    resolve({ id: this.lastID });
                }
            });
        });
    }

    get(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.get(sql, params, (err, result) => {
                if (err) {
                    console.log('Error running sql: ' + sql);
                    console.log(err);
                    reject(err);
                } else {
                    resolve(result);
                }
            });
        });
    }

    all(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.all(sql, params, (err, rows) => {
                if (err) {
                    console.log('Error running sql: ' + sql);
                    console.log(err);
                    reject(err);
                } else {
                    resolve(rows);
                }
            });
        });
    }

    delete(sql, params = []) {
        return new Promise((resolve, reject) => {
            this.db.run(sql, params, function (err) {
                if (err) {
                    console.log('Error running sql ' + sql);
                    console.log(err);
                    reject(err);
                } else {
                    resolve({ rowsDeleted: this.changes });
                }
            });
        });
    }
}

module.exports = Database;;