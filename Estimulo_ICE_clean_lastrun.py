#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on julho 30, 2025, at 12:47
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_9
import csv
import os
from psychopy import event


# Run 'Before Experiment' code from codigo_key


# Run 'Before Experiment' code from codigo_key_2


# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'Estimulo_ICE_clean'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Julio\\Desktop\\Estimulo_Kerstin\\Estimulo_ICE_clean_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_1') is None:
        # initialise key_resp_1
        key_resp_1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_1',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Input" ---
    # Run 'Begin Experiment' code from code_9
    
    
    textbox = visual.TextBox2(
         win, text='Durante o experimento fixe sua visão no ponto vermelho no centro da tela e indique a direção que os itens estão se movendo usando o teclado.\n\n\nPara começar o experimento digite quantas repetições vão ser.', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0.05), draggable=False,      letterHeight=0.05,
         size=(1, 1), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox',
         depth=-1, autoLog=True,
    )
    image_16 = visual.ImageStim(
        win=win,
        name='image_16', 
        image='Logo.png', mask=None, anchor='bottom-right',
        ori=0.0, pos=(0.6, -0.45), draggable=True, size=(0.6,0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "show" ---
    image_15 = visual.ImageStim(
        win=win,
        name='image_15', 
        image='Logo.png', mask=None, anchor='bottom-right',
        ori=0.0, pos=(0.6, -0.45), draggable=True, size=(0.6,0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    text_3 = visual.TextStim(win=win, name='text_3',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='',
        font='Arial',
        pos=(-0.11, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_5 = visual.TextStim(win=win, name='text_5',
        text='',
        font='Arial',
        pos=(0.14, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "trial" ---
    image = visual.ImageStim(
        win=win,
        name='image', units='cm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=0.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', units='cm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-1.0)
    polygon = visual.ShapeStim(
        win=win, name='polygon',
        size=(0.005, 0.005), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, 1.0000, -0.0039], fillColor=[-1.0000, 1.0000, -0.0039],
        opacity=None, depth=-2.0, interpolate=True)
    key_resp_1 = keyboard.Keyboard(deviceName='key_resp_1')
    
    # --- Initialize components for Routine "trial_2" ---
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', units='cm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=0.0)
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', units='cm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-1.0)
    polygon_2 = visual.ShapeStim(
        win=win, name='polygon_2',
        size=(0.005, 0.005), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, 1.0000, -0.0039], fillColor=[-1.0000, 1.0000, -0.0039],
        opacity=None, depth=-2.0, interpolate=True)
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "codigo" ---
    
    # --- Initialize components for Routine "_intertrial_interval_" ---
    polygon_5 = visual.ShapeStim(
        win=win, name='polygon_5',
        size=(0.005, 0.005), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor='red',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "clean" ---
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Input" ---
    # create an object to store info about Routine Input
    Input = data.Routine(
        name='Input',
        components=[textbox, image_16],
    )
    Input.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_9
    tecla_inicial = ''
    
    
    textbox.reset()
    # store start times for Input
    Input.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Input.tStart = globalClock.getTime(format='float')
    Input.status = STARTED
    thisExp.addData('Input.started', Input.tStart)
    Input.maxDuration = None
    # keep track of which components have finished
    InputComponents = Input.components
    for thisComponent in Input.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Input" ---
    Input.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code_9
        keys = event.getKeys()
        
        for key in keys:
            if key == 'return':
                continueRoutine = False
            elif key == 'backspace':
                tecla_inicial = tecla_inicial[:-1]
            elif len(key) == 1:
                tecla_inicial += key
        
        
        # *textbox* updates
        
        # if textbox is starting this frame...
        if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox.frameNStart = frameN  # exact frame index
            textbox.tStart = t  # local t and not account for scr refresh
            textbox.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox.started')
            # update status
            textbox.status = STARTED
            textbox.setAutoDraw(True)
        
        # if textbox is active this frame...
        if textbox.status == STARTED:
            # update params
            pass
        
        # *image_16* updates
        
        # if image_16 is starting this frame...
        if image_16.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            image_16.frameNStart = frameN  # exact frame index
            image_16.tStart = t  # local t and not account for scr refresh
            image_16.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_16, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_16.started')
            # update status
            image_16.status = STARTED
            image_16.setAutoDraw(True)
        
        # if image_16 is active this frame...
        if image_16.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Input.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Input.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Input" ---
    for thisComponent in Input.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Input
    Input.tStop = globalClock.getTime(format='float')
    Input.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Input.stopped', Input.tStop)
    # Run 'End Routine' code from code_9
    thisExp.addData('tecla_inicial', tecla_inicial)
    
    thisExp.addData('textbox.text',textbox.text)
    thisExp.nextEntry()
    # the Routine "Input" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "show" ---
    # create an object to store info about Routine show
    show = data.Routine(
        name='show',
        components=[image_15, text_3, text_4, text_5],
    )
    show.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_3.setText(tecla_inicial)
    text_4.setText('Vão ser ')
    text_5.setText('repetições')
    # store start times for show
    show.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    show.tStart = globalClock.getTime(format='float')
    show.status = STARTED
    thisExp.addData('show.started', show.tStart)
    show.maxDuration = None
    # keep track of which components have finished
    showComponents = show.components
    for thisComponent in show.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "show" ---
    show.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_15* updates
        
        # if image_15 is starting this frame...
        if image_15.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            image_15.frameNStart = frameN  # exact frame index
            image_15.tStart = t  # local t and not account for scr refresh
            image_15.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_15, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_15.started')
            # update status
            image_15.status = STARTED
            image_15.setAutoDraw(True)
        
        # if image_15 is active this frame...
        if image_15.status == STARTED:
            # update params
            pass
        
        # if image_15 is stopping this frame...
        if image_15.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_15.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                image_15.tStop = t  # not accounting for scr refresh
                image_15.tStopRefresh = tThisFlipGlobal  # on global time
                image_15.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_15.stopped')
                # update status
                image_15.status = FINISHED
                image_15.setAutoDraw(False)
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # if text_3 is stopping this frame...
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.tStopRefresh = tThisFlipGlobal  # on global time
                text_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.stopped')
                # update status
                text_3.status = FINISHED
                text_3.setAutoDraw(False)
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # if text_4 is stopping this frame...
        if text_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_4.tStop = t  # not accounting for scr refresh
                text_4.tStopRefresh = tThisFlipGlobal  # on global time
                text_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.stopped')
                # update status
                text_4.status = FINISHED
                text_4.setAutoDraw(False)
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_5.started')
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # if text_5 is stopping this frame...
        if text_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_5.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_5.tStop = t  # not accounting for scr refresh
                text_5.tStopRefresh = tThisFlipGlobal  # on global time
                text_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.stopped')
                # update status
                text_5.status = FINISHED
                text_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            show.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in show.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "show" ---
    for thisComponent in show.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for show
    show.tStop = globalClock.getTime(format='float')
    show.tStopRefresh = tThisFlipGlobal
    thisExp.addData('show.stopped', show.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if show.maxDurationReached:
        routineTimer.addTime(-show.maxDuration)
    elif show.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    MainLoop = data.TrialHandler2(
        name='MainLoop',
        nReps=tecla_inicial, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Estimulos.csv'), 
        seed=None, 
    )
    thisExp.addLoop(MainLoop)  # add the loop to the experiment
    thisMainLoop = MainLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMainLoop.rgb)
    if thisMainLoop != None:
        for paramName in thisMainLoop:
            globals()[paramName] = thisMainLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisMainLoop in MainLoop:
        currentLoop = MainLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisMainLoop.rgb)
        if thisMainLoop != None:
            for paramName in thisMainLoop:
                globals()[paramName] = thisMainLoop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        loop_1 = data.TrialHandler2(
            name='loop_1',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(Arquivo), 
            seed=None, 
        )
        thisExp.addLoop(loop_1)  # add the loop to the experiment
        thisLoop_1 = loop_1.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_1.rgb)
        if thisLoop_1 != None:
            for paramName in thisLoop_1:
                globals()[paramName] = thisLoop_1[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_1 in loop_1:
            currentLoop = loop_1
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_1.rgb)
            if thisLoop_1 != None:
                for paramName in thisLoop_1:
                    globals()[paramName] = thisLoop_1[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[image, image_2, polygon, key_resp_1],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image.setPos((px1, py1))
            image.setImage(imagem)
            image_2.setPos((px2, py2))
            image_2.setImage(imagem)
            # Run 'Begin Routine' code from codigo_key
            loop_start_time = core.getTime()  # Registra o tempo do início da repetição
            
            # Inicializa variável local pra tecla
            tecla_pressionada = ''
            
            # create starting attributes for key_resp_1
            key_resp_1.keys = []
            key_resp_1.rt = []
            _key_resp_1_allKeys = []
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(loop_1, data.TrialHandler2) and thisLoop_1.thisN != loop_1.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.25:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image* updates
                
                # if image is starting this frame...
                if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image.frameNStart = frameN  # exact frame index
                    image.tStart = t  # local t and not account for scr refresh
                    image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.started')
                    # update status
                    image.status = STARTED
                    image.setAutoDraw(True)
                
                # if image is active this frame...
                if image.status == STARTED:
                    # update params
                    pass
                
                # if image is stopping this frame...
                if image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        image.tStop = t  # not accounting for scr refresh
                        image.tStopRefresh = tThisFlipGlobal  # on global time
                        image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.stopped')
                        # update status
                        image.status = FINISHED
                        image.setAutoDraw(False)
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # if image_2 is stopping this frame...
                if image_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_2.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        image_2.tStop = t  # not accounting for scr refresh
                        image_2.tStopRefresh = tThisFlipGlobal  # on global time
                        image_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_2.stopped')
                        # update status
                        image_2.status = FINISHED
                        image_2.setAutoDraw(False)
                
                # *polygon* updates
                
                # if polygon is starting this frame...
                if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon.frameNStart = frameN  # exact frame index
                    polygon.tStart = t  # local t and not account for scr refresh
                    polygon.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.started')
                    # update status
                    polygon.status = STARTED
                    polygon.setAutoDraw(True)
                
                # if polygon is active this frame...
                if polygon.status == STARTED:
                    # update params
                    pass
                
                # if polygon is stopping this frame...
                if polygon.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon.tStop = t  # not accounting for scr refresh
                        polygon.tStopRefresh = tThisFlipGlobal  # on global time
                        polygon.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon.stopped')
                        # update status
                        polygon.status = FINISHED
                        polygon.setAutoDraw(False)
                # Run 'Each Frame' code from codigo_key
                # Captura teclas a cada frame
                theseKeys = event.getKeys(keyList=['up', 'right'], timeStamped=globalClock)
                
                if theseKeys:
                    # Pega só a primeira tecla pra registrar
                    tecla_pressionada = theseKeys[0][0]
                
                
                # *key_resp_1* updates
                waitOnFlip = False
                
                # if key_resp_1 is starting this frame...
                if key_resp_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_1.frameNStart = frameN  # exact frame index
                    key_resp_1.tStart = t  # local t and not account for scr refresh
                    key_resp_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_1.started')
                    # update status
                    key_resp_1.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_1.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_1 is stopping this frame...
                if key_resp_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_1.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_1.tStop = t  # not accounting for scr refresh
                        key_resp_1.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_1.stopped')
                        # update status
                        key_resp_1.status = FINISHED
                        key_resp_1.status = FINISHED
                if key_resp_1.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_1.getKeys(keyList=['left','right','up', 'down'], ignoreKeys=["escape"], waitRelease=True)
                    _key_resp_1_allKeys.extend(theseKeys)
                    if len(_key_resp_1_allKeys):
                        key_resp_1.keys = [key.name for key in _key_resp_1_allKeys]  # storing all keys
                        key_resp_1.rt = [key.rt for key in _key_resp_1_allKeys]
                        key_resp_1.duration = [key.duration for key in _key_resp_1_allKeys]
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # Run 'End Routine' code from codigo_key
            # Salva a coluna 'tecla_final' no .csv oficial
            thisExp.addData('tecla_final', tecla_pressionada)
            
            # check responses
            if key_resp_1.keys in ['', [], None]:  # No response was made
                key_resp_1.keys = None
            loop_1.addData('key_resp_1.keys',key_resp_1.keys)
            if key_resp_1.keys != None:  # we had a response
                loop_1.addData('key_resp_1.rt', key_resp_1.rt)
                loop_1.addData('key_resp_1.duration', key_resp_1.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.250000)
            
            # --- Prepare to start Routine "trial_2" ---
            # create an object to store info about Routine trial_2
            trial_2 = data.Routine(
                name='trial_2',
                components=[image_3, image_4, polygon_2, key_resp_2],
            )
            trial_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_3.setPos((px3, py3))
            image_3.setImage(imagem)
            image_4.setPos((px4,py4))
            image_4.setImage(imagem)
            # Run 'Begin Routine' code from codigo_key_2
            loop_start_time = core.getTime()  # Registra o tempo do início da repetição
            
            
            # Inicializa variável local pra tecla
            tecla_pressionada = ''
            # create starting attributes for key_resp_2
            key_resp_2.keys = []
            key_resp_2.rt = []
            _key_resp_2_allKeys = []
            # store start times for trial_2
            trial_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_2.tStart = globalClock.getTime(format='float')
            trial_2.status = STARTED
            thisExp.addData('trial_2.started', trial_2.tStart)
            trial_2.maxDuration = None
            # keep track of which components have finished
            trial_2Components = trial_2.components
            for thisComponent in trial_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_2" ---
            # if trial has changed, end Routine now
            if isinstance(loop_1, data.TrialHandler2) and thisLoop_1.thisN != loop_1.thisTrial.thisN:
                continueRoutine = False
            trial_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.25:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_3* updates
                
                # if image_3 is starting this frame...
                if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_3.frameNStart = frameN  # exact frame index
                    image_3.tStart = t  # local t and not account for scr refresh
                    image_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.started')
                    # update status
                    image_3.status = STARTED
                    image_3.setAutoDraw(True)
                
                # if image_3 is active this frame...
                if image_3.status == STARTED:
                    # update params
                    pass
                
                # if image_3 is stopping this frame...
                if image_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_3.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        image_3.tStop = t  # not accounting for scr refresh
                        image_3.tStopRefresh = tThisFlipGlobal  # on global time
                        image_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_3.stopped')
                        # update status
                        image_3.status = FINISHED
                        image_3.setAutoDraw(False)
                
                # *image_4* updates
                
                # if image_4 is starting this frame...
                if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_4.frameNStart = frameN  # exact frame index
                    image_4.tStart = t  # local t and not account for scr refresh
                    image_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_4.started')
                    # update status
                    image_4.status = STARTED
                    image_4.setAutoDraw(True)
                
                # if image_4 is active this frame...
                if image_4.status == STARTED:
                    # update params
                    pass
                
                # if image_4 is stopping this frame...
                if image_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_4.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        image_4.tStop = t  # not accounting for scr refresh
                        image_4.tStopRefresh = tThisFlipGlobal  # on global time
                        image_4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_4.stopped')
                        # update status
                        image_4.status = FINISHED
                        image_4.setAutoDraw(False)
                
                # *polygon_2* updates
                
                # if polygon_2 is starting this frame...
                if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon_2.frameNStart = frameN  # exact frame index
                    polygon_2.tStart = t  # local t and not account for scr refresh
                    polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_2.started')
                    # update status
                    polygon_2.status = STARTED
                    polygon_2.setAutoDraw(True)
                
                # if polygon_2 is active this frame...
                if polygon_2.status == STARTED:
                    # update params
                    pass
                
                # if polygon_2 is stopping this frame...
                if polygon_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon_2.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon_2.tStop = t  # not accounting for scr refresh
                        polygon_2.tStopRefresh = tThisFlipGlobal  # on global time
                        polygon_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon_2.stopped')
                        # update status
                        polygon_2.status = FINISHED
                        polygon_2.setAutoDraw(False)
                # Run 'Each Frame' code from codigo_key_2
                # Captura teclas a cada frame
                theseKeys = event.getKeys(keyList=['up', 'right'], timeStamped=globalClock)
                
                if theseKeys:
                    # Pega só a primeira tecla pra registrar
                    tecla_pressionada = theseKeys[0][0]
                
                
                # *key_resp_2* updates
                waitOnFlip = False
                
                # if key_resp_2 is starting this frame...
                if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_2.frameNStart = frameN  # exact frame index
                    key_resp_2.tStart = t  # local t and not account for scr refresh
                    key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_2.started')
                    # update status
                    key_resp_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_2 is stopping this frame...
                if key_resp_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_2.tStartRefresh + 0.25-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_2.tStop = t  # not accounting for scr refresh
                        key_resp_2.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_2.stopped')
                        # update status
                        key_resp_2.status = FINISHED
                        key_resp_2.status = FINISHED
                if key_resp_2.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_2.getKeys(keyList=['left','right','up', 'down'], ignoreKeys=["escape"], waitRelease=True)
                    _key_resp_2_allKeys.extend(theseKeys)
                    if len(_key_resp_2_allKeys):
                        key_resp_2.keys = [key.name for key in _key_resp_2_allKeys]  # storing all keys
                        key_resp_2.rt = [key.rt for key in _key_resp_2_allKeys]
                        key_resp_2.duration = [key.duration for key in _key_resp_2_allKeys]
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_2" ---
            for thisComponent in trial_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_2
            trial_2.tStop = globalClock.getTime(format='float')
            trial_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_2.stopped', trial_2.tStop)
            # Run 'End Routine' code from codigo_key_2
            # Salva a coluna 'tecla_final' no .csv oficial
            thisExp.addData('tecla_final', tecla_pressionada)
            
            # check responses
            if key_resp_2.keys in ['', [], None]:  # No response was made
                key_resp_2.keys = None
            loop_1.addData('key_resp_2.keys',key_resp_2.keys)
            if key_resp_2.keys != None:  # we had a response
                loop_1.addData('key_resp_2.rt', key_resp_2.rt)
                loop_1.addData('key_resp_2.duration', key_resp_2.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_2.maxDurationReached:
                routineTimer.addTime(-trial_2.maxDuration)
            elif trial_2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.250000)
            
            # --- Prepare to start Routine "codigo" ---
            # create an object to store info about Routine codigo
            codigo = data.Routine(
                name='codigo',
                components=[],
            )
            codigo.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for codigo
            codigo.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            codigo.tStart = globalClock.getTime(format='float')
            codigo.status = STARTED
            thisExp.addData('codigo.started', codigo.tStart)
            codigo.maxDuration = None
            # keep track of which components have finished
            codigoComponents = codigo.components
            for thisComponent in codigo.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "codigo" ---
            # if trial has changed, end Routine now
            if isinstance(loop_1, data.TrialHandler2) and thisLoop_1.thisN != loop_1.thisTrial.thisN:
                continueRoutine = False
            codigo.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    codigo.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in codigo.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "codigo" ---
            for thisComponent in codigo.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for codigo
            codigo.tStop = globalClock.getTime(format='float')
            codigo.tStopRefresh = tThisFlipGlobal
            thisExp.addData('codigo.stopped', codigo.tStop)
            # Run 'End Routine' code from code
            # === Junta as respostas ===
            resposta1 = key_resp_1.keys
            resposta2 = key_resp_2.keys
            
            # Define lógica de fusão
            if resposta1:
                resposta_final = resposta1
            elif resposta2:
                resposta_final = resposta2
            else:
                resposta_final = 'None'
            
            # Adiciona uma nova coluna chamada 'resposta_fundida'
            thisExp.addData('Respostas', resposta_final)
            
            # the Routine "codigo" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'loop_1'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "_intertrial_interval_" ---
        # create an object to store info about Routine _intertrial_interval_
        _intertrial_interval_ = data.Routine(
            name='_intertrial_interval_',
            components=[polygon_5],
        )
        _intertrial_interval_.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for _intertrial_interval_
        _intertrial_interval_.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        _intertrial_interval_.tStart = globalClock.getTime(format='float')
        _intertrial_interval_.status = STARTED
        thisExp.addData('_intertrial_interval_.started', _intertrial_interval_.tStart)
        _intertrial_interval_.maxDuration = None
        # keep track of which components have finished
        _intertrial_interval_Components = _intertrial_interval_.components
        for thisComponent in _intertrial_interval_.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "_intertrial_interval_" ---
        # if trial has changed, end Routine now
        if isinstance(MainLoop, data.TrialHandler2) and thisMainLoop.thisN != MainLoop.thisTrial.thisN:
            continueRoutine = False
        _intertrial_interval_.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon_5* updates
            
            # if polygon_5 is starting this frame...
            if polygon_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_5.frameNStart = frameN  # exact frame index
                polygon_5.tStart = t  # local t and not account for scr refresh
                polygon_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_5.started')
                # update status
                polygon_5.status = STARTED
                polygon_5.setAutoDraw(True)
            
            # if polygon_5 is active this frame...
            if polygon_5.status == STARTED:
                # update params
                pass
            
            # if polygon_5 is stopping this frame...
            if polygon_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_5.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_5.tStop = t  # not accounting for scr refresh
                    polygon_5.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_5.stopped')
                    # update status
                    polygon_5.status = FINISHED
                    polygon_5.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                _intertrial_interval_.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in _intertrial_interval_.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "_intertrial_interval_" ---
        for thisComponent in _intertrial_interval_.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for _intertrial_interval_
        _intertrial_interval_.tStop = globalClock.getTime(format='float')
        _intertrial_interval_.tStopRefresh = tThisFlipGlobal
        thisExp.addData('_intertrial_interval_.stopped', _intertrial_interval_.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if _intertrial_interval_.maxDurationReached:
            routineTimer.addTime(-_intertrial_interval_.maxDuration)
        elif _intertrial_interval_.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        thisExp.nextEntry()
        
    # completed tecla_inicial repeats of 'MainLoop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "clean" ---
    # create an object to store info about Routine clean
    clean = data.Routine(
        name='clean',
        components=[],
    )
    clean.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for clean
    clean.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clean.tStart = globalClock.getTime(format='float')
    clean.status = STARTED
    thisExp.addData('clean.started', clean.tStart)
    clean.maxDuration = None
    # keep track of which components have finished
    cleanComponents = clean.components
    for thisComponent in clean.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "clean" ---
    clean.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            clean.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in clean.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clean" ---
    for thisComponent in clean.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clean
    clean.tStop = globalClock.getTime(format='float')
    clean.tStopRefresh = tThisFlipGlobal
    thisExp.addData('clean.stopped', clean.tStop)
    thisExp.nextEntry()
    # the Routine "clean" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from Remove_notes
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
